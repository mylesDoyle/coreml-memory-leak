#import "Benchmark.h"
#include <string>
#include <vector>
#include "torch/script.h"

#include <CoreML/CoreML.h>

#include "torch/library.h"

#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"

static std::string model_path = "";
static bool coreml_model;
static std::vector<std::string> model_inputs;
static std::string input_dims = "";
static std::string input_type = "float";
static BOOL print_output = false;
static int warmup = 10;
static int iter = 10;


@interface Converter : NSObject

+ (at::Tensor) multiArrayToTensor:(MLMultiArray*) input;

@end

@implementation Converter

+ (at::Tensor) multiArrayToTensor:(MLMultiArray*) input {
  
  float* input_ptr = (float *) input.dataPointer;
  NSArray* shape = input.shape;
  int batch = [shape[0] intValue];
  int ch = [shape[1] intValue];
  int height = [shape[2] intValue];
  int width = [shape[3] intValue];
  int pixels = ch * height * width;
  
  at::Tensor output = at::ones({batch, ch, height, width});
  float* output_ptr = output.data_ptr<float>();
  
  for (int pixel_index = 0; pixel_index < pixels; ++pixel_index) {
    output_ptr[pixel_index] = input_ptr[pixel_index];
  }
  
  return output;
}

+ (MLMultiArray*) tensorToMultiArray:(at::Tensor) input {
  
  float* input_ptr = input.data_ptr<float>();
  int batch = (int) input.size(0);
  int ch = (int) input.size(1);
  int height = (int) input.size(2);
  int width = (int) input.size(3);
  
  int pixels = ch * height * width;

  NSArray* shape = @[[NSNumber numberWithInt:batch], [NSNumber numberWithInt: ch], [NSNumber numberWithInt: height], [NSNumber numberWithInt: width]];

  MLMultiArray* output = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:NULL];
  float* output_ptr = (float *) output.dataPointer;

  for (int pixel_index = 0; pixel_index < pixels; ++pixel_index) {
    output_ptr[pixel_index] = input_ptr[pixel_index];
  }
  
  return output;
}

@end

@implementation Benchmark

+ (BOOL)setup:(NSDictionary*)config {
  NSString* modelPath = @"~/Library/model.pt";
  modelPath = [modelPath stringByExpandingTildeInPath];
  //std::cout << std::string(modelPath.UTF8String) << std::endl;
  //NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"ml_model" ofType:@"pt"]; // private/var/containers/Bundle/Application/93C39ECD-BD09-462C-9B64-2928BFF6C54C/TestApp.app/model.pt
  if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
    NSLog(@"model file doesn't exist!");
    return NO;
  }
  model_path = std::string(modelPath.UTF8String);
  std::cout << model_path << std::endl;

  
 // Model input paths
  for (NSString* str in ((NSArray*)config[@"model_inputs"])) {
    NSString* inputPath = [str stringByExpandingTildeInPath];
    std::string cppStr = std::string([inputPath UTF8String]);
    model_inputs.push_back(cppStr);
  }
  
  if (config[@"input_dims"]) {
    input_dims = std::string(((NSString*)config[@"input_dims"]).UTF8String);
  }
  
  input_type = std::string(((NSString*)config[@"input_type"]).UTF8String);
  warmup = ((NSNumber*)config[@"warmup"]).intValue;
  iter = ((NSNumber*)config[@"iter"]).intValue;
  coreml_model = ((NSNumber*)config[@"coreml_model"]).boolValue;
  print_output = NO;
  return YES;
}

+ (NSString*)run {
  std::vector<std::string> logs;
#define UI_LOG(fmt, ...)                                          \
  {                                                               \
    NSString* log = [NSString stringWithFormat:fmt, __VA_ARGS__]; \
    NSLog(@"%@", log);                                            \
    logs.push_back(log.UTF8String);                               \
  }

  CAFFE_ENFORCE_GE(input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(input_type.size(), 0, "Input type must be specified.");
  
  std::vector<std::string> input_dims_list;
  std::vector<std::string> input_type_list = caffe2::split(';', input_type);
  
  std::vector<c10::IValue> inputs;
  if (input_dims.size() > 0) {
    std::vector<std::string> input_dims_list = caffe2::split(';', input_dims);
    CAFFE_ENFORCE_EQ(input_dims_list.size(), input_type_list.size(),
                     "Input dims and type should have the same number of items.");
    for (size_t i = 0; i < input_dims_list.size(); ++i) {
      auto input_dims_str = caffe2::split(',', input_dims_list[i]);
      std::vector<int64_t> input_dims;
      for (const auto& s : input_dims_str) {
        input_dims.push_back(c10::stoi(s));
      }
      if (input_type_list[i] == "float") {
        inputs.push_back(torch::ones(input_dims, at::ScalarType::Float));
      } else if (input_type_list[i] == "uint8_t") {
        inputs.push_back(torch::ones(input_dims, at::ScalarType::Byte));
      } else {
        CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < model_inputs.size(); ++i) {
      std::string attr = "value";
      std::string or_else = "error";
      torch::jit::mobile::Module module = torch::jit::_load_for_mobile(model_inputs[i]);
      
      c10::IValue tensor = module.attr(attr, or_else).toIValue();
      inputs.push_back(tensor);
    }
  }
  
  c10::InferenceMode mode;
  if (!coreml_model) {
    
    //////////////////////
    // CPU model
    //////////////////////
    
    auto module = torch::jit::_load_for_mobile(model_path);
    
    UI_LOG(@"Running warmup runs", nil);
    CAFFE_ENFORCE(warmup >= 0, "Number of warm up runs should be non negative, provided ", warmup,
                  ".");
    for (int i = 0; i < warmup; ++i) {
      module.forward(inputs);
    }
    UI_LOG(@"Main runs", nil);
    CAFFE_ENFORCE(iter >= 0, "Number of main runs should be non negative, provided ", iter, ".");
      
    caffe2::Timer timer;
    auto millis = timer.MilliSeconds();
    for (int i = 0; i < iter; ++i) {
      module.forward(inputs);
    }
    millis = timer.MilliSeconds();
    
    // Save result of last iteration
    c10::IValue result = module.forward(inputs);
    auto bytes = torch::jit::pickle_save(result);
    std::string save_path = std::string([[@"~/Library/mobile_result.zip" stringByExpandingTildeInPath] UTF8String]);
    std::ofstream fout(save_path, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
    
    UI_LOG(@"Main run finished. Milliseconds per iter: %.3f", millis / iter, nil);
    UI_LOG(@"Iters per second: : %.3f", 1000.0 * iter / millis, nil);
    UI_LOG(@"Done.", nil);
      
    exit(0);

  } else {
    
    /////////////////////////////
    // CoreML model
    /////////////////////////////
    
    NSError* __autoreleasing __nullable* __nullable error = nil;

    NSString* modelPath = [NSString stringWithUTF8String:model_path.c_str()];
    NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
    NSURL* compiledModel = [MLModel compileModelAtURL:modelURL error:error];
    
    MLModel* module = [MLModel modelWithContentsOfURL:compiledModel error:NULL];
    
    NSMutableDictionary* feature_inputs = [[NSMutableDictionary alloc] init];
    for (int i = 0; i < inputs.size(); ++i) {
      NSString* key = [NSString stringWithFormat:@"input_%d", i];
      [feature_inputs setValue:[Converter tensorToMultiArray: inputs[i].toTensor()] forKey: key];
    }
    MLDictionaryFeatureProvider* feature_provider = [[[MLDictionaryFeatureProvider alloc] init] initWithDictionary:feature_inputs error:NULL];
  
    for (int i = 0; i < warmup; ++i) {
      [module predictionFromFeatures:feature_provider error:error];
    }
    
    caffe2::Timer timer;
    auto millis = timer.MilliSeconds();
    for (int i = 0; i < iter; ++i) {
      [module predictionFromFeatures:feature_provider error:NULL];
    }
    millis = timer.MilliSeconds();
        
    c10::Dict<std::string, at::Tensor> result_map;
    
    id<MLFeatureProvider> output_feature_provider = [module predictionFromFeatures:feature_provider error:NULL];
    for (NSString* key in [output_feature_provider featureNames]) {
      at::Tensor value = [Converter multiArrayToTensor: [[output_feature_provider featureValueForName: key] multiArrayValue]];
      result_map.insert(std::string([key UTF8String]), value);
    }
    
    auto bytes = torch::jit::pickle_save(result_map);
    std::string save_path = std::string([[@"~/Library/mobile_result.zip" stringByExpandingTildeInPath] UTF8String]);
    std::ofstream fout(save_path, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
    
    UI_LOG(@"Main run finished. Milliseconds per iter: %.3f", millis / iter, nil);
    UI_LOG(@"Iters per second: : %.3f", 1000.0 * iter / millis, nil);
    UI_LOG(@"Done.", nil);
        
    exit(0);
    
  }
}

@end

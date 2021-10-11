#import "ViewController.h"
#import <torch/script.h>
#import "Benchmark.h"

@interface ViewController ()
@property(weak, nonatomic) IBOutlet UITextView* textView;

@end

@implementation ViewController {
}

- (void)viewDidLoad {
  [super viewDidLoad];
    
  NSDictionary* env = [[NSProcessInfo processInfo] environment];
    
  NSArray* model_inputs = NULL;
  NSString* model_inputs_string = [env objectForKey:@"model_inputs"];
  if (model_inputs_string) {
      model_inputs = [model_inputs_string componentsSeparatedByString:@";"];
  }
  
  NSDictionary* config;
  
  if ([env objectForKey:@"input_dims"]) {
   config = @{
     @"coreml_model": [env objectForKey:@"coreml_model"],
     @"input_dims": [env objectForKey:@"input_dims"],
     @"input_type": [env objectForKey:@"input_type"],
     @"warmup": [env objectForKey:@"warmup"],
     @"iter": [env objectForKey:@"iter"]
    };
  } else {
    config = @{
      @"coreml_model": [env objectForKey:@"coreml_model"],
      @"model_inputs": model_inputs,
      @"input_type": [env objectForKey:@"input_type"],
      @"warmup": [env objectForKey:@"warmup"],
      @"iter": [env objectForKey:@"iter"]
    };
  }


    
  self.textView.text = [config description];

  [Benchmark setup:config];
  [self runBenchmark];
}

- (void)runBenchmark {
//self.textView.text += @"Start benchmarking...\n";
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSString* text = [Benchmark run];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.textView.text = [self.textView.text stringByAppendingString:text];
    });
  });
}

- (IBAction)reRun:(id)sender {
  self.textView.text = @"";
  dispatch_async(dispatch_get_main_queue(), ^{
    [self runBenchmark];
  });
}

@end

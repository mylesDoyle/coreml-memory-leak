import coremltools
import torch
import torch.nn as nn

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		upscale_factor  = 8

		self.Conv1 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, stride = 1)
		self.Conv2 = nn.Conv2d(48, 48, 3, 1)
		self.Conv3 = nn.Conv2d(48, 3 * (upscale_factor*upscale_factor), 3, 1)

		self.PS = nn.PixelShuffle(upscale_factor)

	def forward(self, x):

		Conv1 = self.Conv1(x)
		Conv2 = self.Conv2(Conv1)
		Conv3 = self.Conv3(Conv2)
		y 	  = self.PS(Conv3)

		return y

def convert_torch_to_coreml(torch_model, input_shapes, save_path):

	torchscript_model = torch.jit.script(torch_model)

	mlmodel = coremltools.converters.convert(
		torchscript_model,
		inputs=[coremltools.TensorType(name=f'input_{i}', shape=input_shape) for i, input_shape in enumerate(input_shapes)],
	)

	mlmodel.save(save_path)

if __name__ == "__main__":
	torch_model = Model()

	# input_shapes = [[1,48,256,135]]	# 2K
	input_shapes = [[1,48,480,270]]		# 4K
	coreml_model_path = "./toy.mlmodel"
	
	convert_torch_to_coreml(torch_model, input_shapes, coreml_model_path)
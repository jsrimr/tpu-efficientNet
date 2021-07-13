import efficientnet_builder
from condconv import efficientnet_condconv_builder  # 와.. if 문 쓰려면 operation 을 정의해야 하는구나..
# from tpu import efficientnet_x_builder


def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""

  if model_name.startswith('efficientnet-condconv-'):
    return efficientnet_condconv_builder
  # elif model_name.startswith('efficientnet-x-'):
  #   return efficientnet_x_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu* orefficientnet-condconv*, efficientnet-lite*')

def get_model_input_size(model_name):
  """Get model input size for a given model name."""

  if model_name.startswith('efficientnet-condconv-'):
      _, _, image_size, _, _ = (
          efficientnet_condconv_builder.efficientnet_condconv_params(model_name))
  # elif model_name.startswith('efficientnet-x'):
  #     _, _, image_size, _, _ = efficientnet_x_builder.efficientnet_x_params(
  #         model_name)
  elif model_name.startswith('efficientnet'):
      _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  else:
      raise ValueError(
          'Model must be either efficientnet-b* or efficientnet-x-b* or efficientnet-edgetpu* or efficientnet-condconv*, efficientnet-lite*')
  return image_size
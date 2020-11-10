Pod::Spec.new do |s|
  s.name             = 'NNToolkitCore'
  s.version          = '0.4.1'
  s.summary          = 'Core C library with NN filters'
  s.description      = "LSTM. Conv1d. GRU. BatchNorm. Dense. Audio processing tools"

  s.homepage         = 'https://github.com/techpro-studio/NNToolkitCore'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Oleksii Moiseenko' => 'oleksiimoiseenko@gmail.com' }
  s.source           = { :git => 'https://github.com/techpro-studio/NNToolkitCore.git', :tag => s.version }

  s.ios.deployment_target = '8.0'
  s.watchos.deployment_target = '4.0'
  s.tvos.deployment_target = '9.0'
  s.osx.deployment_target = '10.9'

  s.public_header_files =
  'nn_toolkit_core/include/*.{h}',
  'nn_toolkit_core/layers/*.{h}',
  'nn_toolkit_core/train/*.{h}',
  'nn_toolkit_core/signal/spectrogram.h',
  'nn_toolkit_core/signal/window.h',
  'nn_toolkit_core/core/debug.h'

  s.source_files =
      'nn_toolkit_core/include/*.h',
      'nn_toolkit_core/core/*.h',
      'nn_toolkit_core/core/debug.c',
      'nn_toolkit_core/core/apple_ops.c',
      'nn_toolkit_core/layers/*.{h,c}',
      'nn_toolkit_core/train/*.{h,c}',
      'nn_toolkit_core/signal/*.{h,c}'


  s.pod_target_xcconfig = {
      'HEADER_SEARCH_PATHS' =>
        '"${PODS_TARGET_SRCROOT}" '
    }
  s.frameworks = 'Accelerate'
   
end

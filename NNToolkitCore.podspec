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
  'nntoolkitcore/include/*.{h}',
  'nntoolkitcore/layers/*.{h}',
  'nntoolkitcore/train/*.{h}',
  'nntoolkitcore/signal/spectrogram.h',
  'nntoolkitcore/signal/window.h',
  'nntoolkitcore/core/debug.h'
  'nntoolkitcore/core/ops.h'

  s.source_files =
      'nntoolkitcore/include/*.h',
      'nntoolkitcore/core/*.h',
      'nntoolkitcore/core/debug.c',
      'nntoolkitcore/core/default_ops.cc',
      'nntoolkitcore/layers/*.{h,c}',
      'nntoolkitcore/train/*.{h,c}',
      'nntoolkitcore/signal/*.{h,c}'
      'thirdparty/pffft/*.{h,c}'


  s.pod_target_xcconfig = {
      'HEADER_SEARCH_PATHS' =>
        '"${PODS_TARGET_SRCROOT}" '
    }
  s.frameworks = 'Accelerate'
   
end

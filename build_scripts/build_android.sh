mkdir cmake-android
# shellcheck disable=SC2164
cd cmake-android
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/alex/Library/Android/sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=armeabi-v7a\
        -DANDROID_ARM_NEON=ON. \
        -DCMAKE_BUILD_TYPE=Release\
        -DANDROID_NATIVE_API_LEVEL=16 ../..
cmake --build .
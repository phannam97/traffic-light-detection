# traffic-light-detection
A state of the art real-time object detection system for Python (Intellij). This project has GPU support, with GPU the detection works much faster. The primary goal of this project is an easy use of yolo, this package is available on nuget and you must only install two packages to start detection. In the background we are use the Windows Yolo version of [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). Send an image path or the byte array to [yolo](https://github.com/pjreddie/darknet) and receive the position of the detected objects. Our project is meant to return the object-type and -position as processable data. This library supports [YoloV3 and YoloV4 Pre-Trained Datasets](#pre-trained-dataset)
## Performance
It is important to use GPU mode for fast object detection. It is also important not to instantiate the wrapper over and over again. A further optimization is to transfer the images as byte stream instead of passing a file path. GPU detection is usually 10 times faster!
It is important to use the mentioned version `10.2`

1) Install the latest Nvidia driver for your graphic device
2) [Install Nvidia CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-downloads) (must be installed add a hardware driver for cuda support)
3) [Download Nvidia cuDNN v7.6.5 for CUDA 10.2](https://developer.nvidia.com/rdp/cudnn-download)
4) Copy the `cudnn64_7.dll` from the output directory of point 2. into the project folder.

## Build requirements
- Intellj (Django)

### GPU
Graphic card | Single precision | Memory | Slot | YOLOv3 | YOLOv4 |
--- | --- | --- | --- | --- | --- |
NVIDIA GeForce GTX 1060 | 4372 GFLOPS | 6 GB | Dual | 100 ms | --- |

### Directory Structure

You should have this files in your program directory.

    .
    ├── yolo_cpp_dll_gpu.dll      # yolo runtime for gpu
    ├── cudnn64_7.dll             # required by yolo_cpp_dll_gpu (optional only required for gpu processig)
    ├── opencv_world340.dll       # required by yolo_cpp_dll_xxx (process image as byte data detect_mat)
    ├── pthreadGC2.dll            # required by yolo_cpp_dll_xxx (POSIX Threads)
    ├── pthreadVC2.dll            # required by yolo_cpp_dll_xxx (POSIX Threads)
    ├── msvcr100.dll              # required by pthread (POSIX Threads)

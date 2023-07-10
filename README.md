# Pupil Ray

Pupil Ray is a rendering methods playground for personal learning, witch is based on the HWRT framework [PupilOptixLab](https://github.com/mchenwang/PupilOptixLab)(OptiX7.5).

## Implemented Algorithms Overview

### megakernel path tracing
A basic path tracing method use OptiX.

### wavefront path tracing

Compared to the megakernel path tracing, wavefront method is more GPU-friendly which means it is more efficient.

Main ideas for implementation:
- split mega kernel into small kernels
- trade space for time (record the current bounce information of the ray path in screen space)



## screenshot

### classroom

![](https://github.com/mchenwang/PupilRay/raw/main/image/classroom.png)

### bathroom

![](https://github.com/mchenwang/PupilRay/raw/main/image/bathroom1.png)

### bathroom2

![](https://github.com/mchenwang/PupilRay/raw/main/image/bathroom2.png)

### bedroom

![](https://github.com/mchenwang/PupilRay/raw/main/image/bedroom.png)

### kitchen

![](https://github.com/mchenwang/PupilRay/raw/main/image/kitchen.png)

### living-room-2

![](https://github.com/mchenwang/PupilRay/raw/main/image/livingroom2.png)

### living-room-3

![](https://github.com/mchenwang/PupilRay/raw/main/image/livingroom3.png)

### staircase

![](https://github.com/mchenwang/PupilRay/raw/main/image/staircase.png)

### staircase2

![](https://github.com/mchenwang/PupilRay/raw/main/image/staircase2.png)

### veach-ajar

![](https://github.com/mchenwang/PupilRay/raw/main/image/veach-ajar.png)

### veach-mis

![](https://github.com/mchenwang/PupilRay/raw/main/image/veach-mis.png)

### lamp

![](https://github.com/mchenwang/PupilRay/raw/main/image/lamp.png)

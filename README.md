# Deep Learning based Signature Detection (YOLOv5x)
## Introduction

In the modern, rapidly developing world, where all spheres of human life are actively digitized, the problem of interaction with a large number of documents is quite acute. Yes, in the XIX and XX century, the accountant's vacancy was considered fashionable and prestigious, because there were very few qualified specialists, and their work was evaluated quite highly. But times are changing, technologies are developing, and people are tired of doing the same type of work on their own. Therefore, by automating this process, it would be possible to save a significant amount of time and qualified resources. This project is part of a large system for obtaining named entities from documents.

## Theory

Initially, I was thinking about developing my own signature detection model. But since this is only a small part of the project and the quality of detecting the boundaries of an object is not as important to me as the fact of its presence. Then I decided to use a ready-made pre-trained model. Among popular architectures, I considered YOLOv5x to be the simplest and most convenient option. If you pay attention to the graph below, you can see that YOLOv5x shows the best accuracy results compared to YOLOv5l, YOLOv5m, YOLOv5s.

![Pipeline]yolo_comparison.png
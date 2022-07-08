﻿// Copyright 2022 Butescu Vladimir

using Traffic_signs_recognition.Yolov5Scorer;

namespace Traffic_signs_recognition.MLModel
{
    public abstract class YoloModel
    {
        public abstract int Width { get; set; }
        public abstract int Height { get; set; }
        public abstract int Depth { get; set; }

        public abstract int Dimensions { get; set; }

        public abstract int[] Strides { get; set; }
        public abstract int[][][] Anchors { get; set; }
        public abstract int[] Shapes { get; set; }

        public abstract float Confidence { get; set; }
        public abstract float MulConfidence { get; set; }
        public abstract float Overlap { get; set; }

        public abstract string[] Outputs { get; set; }
        public abstract List<YoloLabel> Labels { get; set; }
        public abstract bool UseDetect { get; set; }
    }
}

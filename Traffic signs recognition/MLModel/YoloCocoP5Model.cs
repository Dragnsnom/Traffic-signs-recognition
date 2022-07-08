// Copyright 2022 Butescu Vladimir

using Traffic_signs_recognition.Yolov5Scorer;

namespace Traffic_signs_recognition.MLModel
{
    public class YoloCocoP5Model : YoloModel
    {
        public override int Width { get; set; } = 640;
        public override int Height { get; set; } = 640;
        public override int Depth { get; set; } = 3;

        public override int Dimensions { get; set; } = 25;

        public override int[] Strides { get; set; } = new int[] { 8, 16, 32 };

        public override int[][][] Anchors { get; set; } = new int[][][]
        {
            new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
            new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
            new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
        };

        public override int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

        public override float Confidence { get; set; } = 0.20f;
        public override float MulConfidence { get; set; } = 0.25f;
        public override float Overlap { get; set; } = 0.45f;

        public override string[] Outputs { get; set; } = new[] { "output" };

        public override List<YoloLabel> Labels { get; set; } = new List<YoloLabel>()
        {
            new YoloLabel { Id = 1, Name = "100km" },
            new YoloLabel { Id = 2, Name = "20km" },
            new YoloLabel { Id = 3, Name = "30km" },
            new YoloLabel { Id = 4, Name = "40km" },
            new YoloLabel { Id = 5, Name = "50km" },
            new YoloLabel { Id = 6, Name = "60km" },
            new YoloLabel { Id = 7, Name = "80km" },
            new YoloLabel { Id = 8, Name = "Dangerous left turn" },
            new YoloLabel { Id = 9, Name = "Dangerous turn" },
            new YoloLabel { Id = 10, Name = "Give Way" },
            new YoloLabel { Id = 11, Name = "Movement Prohibition" },
            new YoloLabel { Id = 12, Name = "Priotiry Road" },
            new YoloLabel { Id = 13, Name = "Truck traffic is prohibited" },
            new YoloLabel { Id = 14, Name = "crosswalk" },
            new YoloLabel { Id = 15, Name = "move to the left" },
            new YoloLabel { Id = 16, Name = "move to the right" },
            new YoloLabel { Id = 17, Name = "moving straight and left" },
            new YoloLabel { Id = 18, Name = "moving straight and to the right" },
            new YoloLabel { Id = 19, Name = "stop" },
            new YoloLabel { Id = 20, Name = "straight ahead" },

        };

        public override bool UseDetect { get; set; } = true;

    }
}

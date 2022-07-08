// Copyright 2022 Butescu Vladimir

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

using Traffic_signs_recognition.MLModel;

namespace Traffic_signs_recognition.Yolov5Scorer {
    public class YoloScorer<T> : IDisposable where T : YoloModel {

        private readonly T _model;
        private readonly InferenceSession _inferenceSession;
        private float Sigmoid(float value) {
            return 1 / (1 + (float)Math.Exp(-value));
        }

        private float[] Xywh2xyxy(float[] source) {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public float Clamp(float value, float min, float max) {
            return (value < min) ? min : (value > max) ? max : value;
        }

        private Bitmap ResizeImage(Image image)
        {
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(_model.Width, _model.Height, format);

            var (w, h) = (image.Width, image.Height);
            var (xRatio, yRatio) = (_model.Width / (float)w, _model.Height / (float)h);
            var ratio = Math.Min(xRatio, yRatio);
            var (width, height) = ((int)(w * ratio), (int)(h * ratio));
            var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2));
            var roi = new Rectangle(x, y, width, height);

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0));

                graphics.SmoothingMode = SmoothingMode.None;
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.PixelOffsetMode = PixelOffsetMode.Half;

                graphics.DrawImage(image, roi);
            }

            return output;
        }

        private Tensor<float> ExtractPixels(Image image)
        {
            var bitmap = (Bitmap)image;

            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;

            var tensor = new DenseTensor<float>(new[] { 1, 3, _model.Height, _model.Width });

            unsafe
            {
                Parallel.For(0, bitmapData.Height, (y) =>
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);

                    Parallel.For(0, bitmapData.Width, (x) =>
                    {
                        tensor[0, 0, y, x] = row[x * bytesPerPixel + 2] / 255.0F;
                        tensor[0, 1, y, x] = row[x * bytesPerPixel + 1] / 255.0F;
                        tensor[0, 2, y, x] = row[x * bytesPerPixel + 0] / 255.0F;
                    });
                });

                bitmap.UnlockBits(bitmapData);
            }

            return tensor;
        }

        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                resized = ResizeImage(image);
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs);

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs)
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            return output.ToArray();
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); 
            var gain = Math.Min(xGain, yGain); 

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); 
            Parallel.For(0, (int)output.Length / _model.Dimensions, (i) =>
            {
                if (output[0, i, 4] <= _model.Confidence) return; 

                Parallel.For(5, _model.Dimensions, (j) =>
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; 
                });

                Parallel.For(5, _model.Dimensions, (k) =>
                {
                    if (output[0, i, k] <= _model.MulConfidence) return; 

                    float xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain;
                    float yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain;
                    float xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain;
                    float yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain;

                    xMin = Clamp(xMin, 0, w - 0);
                    yMin = Clamp(yMin, 0, h - 0);
                    xMax = Clamp(xMax, 0, w - 1);
                    yMax = Clamp(yMax, 0, h - 1);

                    YoloLabel label = _model.Labels[k - 5];

                    var prediction = new YoloPrediction(label, output[0, i, k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                });
            });

            return result.ToList();
        }

        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            var gain = Math.Min(xGain, yGain);

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);

            Parallel.For(0, output.Length, (i) =>
            {
                int shapes = _model.Shapes[i];

                Parallel.For(0, _model.Anchors[0].Length, (a) =>
                {
                    Parallel.For(0, shapes, (y) =>
                    {
                        Parallel.For(0, shapes, (x) =>
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            if (buffer[4] <= _model.Confidence) return;

                            List<float> scores = buffer.Skip(5).Select(b => b * buffer[4]).ToList();

                            float mulConfidence = scores.Max();

                            if (mulConfidence <= _model.MulConfidence) return;

                            float rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i];
                            float rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i];

                            float rawW = (float)Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0];
                            float rawH = (float)Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1];

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            float xMin = Clamp((xyxy[0] - xPad) / gain, 0, w - 0);
                            float yMin = Clamp((xyxy[1] - yPad) / gain, 0, h - 0);
                            float xMax = Clamp((xyxy[2] - xPad) / gain, 0, w - 1);
                            float yMax = Clamp((xyxy[3] - yPad) / gain, 0, h - 1);

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPrediction(label, mulConfidence)
                            {
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        });
                    });
                });
            });

            return result.ToList();
        }

        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
        }

        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) 
            {
                foreach (var current in result.ToList())
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height;
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea;
                    float overlap = intArea / unionArea;

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        public List<YoloPrediction> Predict(Image image)
        {
            return Supress(ParseOutput(Inference(image), image));
        }
        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }

        public YoloScorer(string weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }

        public YoloScorer(Stream weights, SessionOptions opts = null) : this()
        {
            using (var reader = new BinaryReader(weights))
            {
                _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
            }
        }

        public YoloScorer(byte[] weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}

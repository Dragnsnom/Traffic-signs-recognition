// Copyright 2022 Butescu Vladimir

using DirectShowLib;
using Emgu.CV;
using Emgu.CV.Structure;
using Traffic_signs_recognition.MLModel;
using Traffic_signs_recognition.Yolov5Scorer;

namespace Traffic_signs_recognition {
    public partial class Form1 : Form {

        private VideoCapture capture;
        private string filename;
        private double frame;
        private int frameCount = 0;
        private double fps = 0;
        private bool play = false;

        private DsDevice[] WebCams;
        private bool cameraOn = true;
        private int selectedcameraID = 0;

        public Form1() {
            InitializeComponent();
        }
        private static Image Find(Image image) { 

            using var scorer = new YoloScorer<YoloCocoP5Model>("MLModel/best.onnx");
            List<YoloPrediction> predictions = scorer.Predict(image);
            using var graphics = Graphics.FromImage(image);

            foreach (var prediction in predictions) {

                double score = Math.Round(prediction.Score, 2);

                graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
                    new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

                graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
            }

            return image;
        }
        private  async void ReadFrame() {
            Mat m = new Mat();

            while(play && frameCount < frame - 1) {

                frameCount += 1;
                capture.Set(Emgu.CV.CvEnum.CapProp.PosFrames, frameCount);
                capture.Read(m);


                fps = capture.Get(Emgu.CV.CvEnum.CapProp.Fps);
               // pictureBox1.Image = m.ToBitmap();
                pictureBox1.Image = Find(m.ToBitmap());
                toolStripLabel1.Text = $"{frameCount} / {frame}";
                toolStripLabel2.Text = $"FPS: {fps, 0:F1}";

                await Task.Delay(1000 / Convert.ToInt16(fps));
            }
        }
        private void toolStripButton1_Click(object sender, EventArgs e) {

            try {
                if (capture == null)
                    throw new Exception("Видео не выбрано");
                if (!play)
                    throw new Exception("Видео и так на паузе");

                play = false;
            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
        private void открытьToolStripMenuItem_Click(object sender, EventArgs e) {

            try {
                OpenFileDialog dialog = new OpenFileDialog();
                DialogResult res = openFileDialog1.ShowDialog();
                dialog.Filter = "Video Files | *.mp4; *.MP4";

                if (res == DialogResult.OK) {
                    capture = new VideoCapture(openFileDialog1.FileName);
                    Mat m = new Mat();

                    capture.Read(m);
                    pictureBox1.Image = m.ToBitmap();

                    fps = capture.Get(Emgu.CV.CvEnum.CapProp.Fps);
                    frame = capture.Get(Emgu.CV.CvEnum.CapProp.FrameCount);
                    frameCount = 1;
 
                }
                else {
                    MessageBox.Show("Видео не было выбрано!", "Error",
                        MessageBoxButtons.OK, MessageBoxIcon.None);
                }



            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "Error", 
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private void toolStripButton2_Click(object sender, EventArgs e) {

            try {

                if (capture == null)
                    throw new Exception("Видео не выбрано");
                if (play)
                    throw new Exception("Видео не на паузе");

                play = true;
                ReadFrame();


            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

        }
        private void открытьФотоToolStripMenuItem_Click(object sender, EventArgs e) {

            try {
                DialogResult res = openFileDialog1.ShowDialog();

                if (res == DialogResult.OK) {

                    filename = openFileDialog1.FileName;
                    //pictureBox1.Image = Image.FromFile(filename);
                    pictureBox1.Image = Find(Image.FromFile(filename));

                }
                else {
                    MessageBox.Show("Картинка не была выбрана!", "Error",
                        MessageBoxButtons.OK, MessageBoxIcon.None);
                }


            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }
        private void Form1_Load(object sender, EventArgs e) {

            WebCams = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);

            for (int i = 0; i < WebCams.Length; i++)
                toolStripComboBox1.Items.Add(WebCams[i].Name); 


        }
        private void toolStripComboBox1_SelectedIndexChanged(object sender, EventArgs e) {
            selectedcameraID = toolStripComboBox1.SelectedIndex;
        }
        private void toolStripButton3_Click(object sender, EventArgs e) {

            try {
                if (cameraOn) {
                    if (WebCams.Length == 0) {
                        throw new Exception("Нет доступных камер");
                    }
                    else {
                        capture = new VideoCapture(selectedcameraID);
                        capture.ImageGrabbed += Capture_ImageGrabbed;
                        capture.Start();

                        toolStripButton3.Text = "выключить камеру";
                        cameraOn = false;
                    }

                }
                else if (!cameraOn) {

                    capture.Stop();
                    capture.Dispose();
                    capture = null;

                    pictureBox1.Image.Dispose();
                    pictureBox1.Image = null;

                    //pictureBox2.Image.Dispose();
                    //pictureBox2.Image = null;


                    toolStripButton3.Text = "включить  камеру";
                    cameraOn = true;

                }

            }
            catch (Exception ex) {

                MessageBox.Show(ex.Message, "Ошибка!", 
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }
        private void Capture_ImageGrabbed(object? sender, EventArgs e) {

            try {

                Mat m = new Mat();
                capture.Retrieve(m);
                //pictureBox1.Image = m.ToImage<Bgr, byte>().Flip(Emgu.CV.CvEnum.FlipType.Horizontal).ToBitmap();
                pictureBox1.Image = Find(m.ToImage<Bgr, byte>().Flip(Emgu.CV.CvEnum.FlipType.Horizontal).ToBitmap());
            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "Ошибка!",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }
    }
}
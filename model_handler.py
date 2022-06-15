import shutil
import subprocess
import sys
import time

import cv2
from PIL import ImageTk, ImageOps
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import visualization_utils as vis_util, label_map_util

from BasicFunctions import *


class ModelHandler:
    def __init__(self, gui, data_handler):
        self.gui = gui
        self.root = gui.root
        self.data_handler = data_handler

        # Training
        self.training_dir = None

        # Testing
        self.category_index = None
        self.loaded_model = None

    def train_model(self):
        """
        Train a new model
        The user chooses the directory for the training files

        :return: None
        """
        self.training_dir = ask_empty_directory(self.root, force=False, initialdir=os.getcwd(), title='Select a directory for the model files')
        if self.training_dir == '':
            self.training_dir = None
            return
        self.__prepare_config()
        PrintUtils.info('A config file was created at {}'.format(os.path.join(self.training_dir, 'pipeline.config')))
        PrintUtils.getinput('Press enter once you have finished customizing your config')

        tb_proc = subprocess.Popen(['tensorboard',
                                    '--logdir={path}/'.format(path=self.training_dir),
                                    '--host=localhost',
                                    # '--port=6006'],
                                    '--port=0'],
                                   env=os.environ.copy())

        time.sleep(15)

        # PrintUtils.info('Tensorboard is open, enter link "http://localhost:6006/"')

        start_time = time.time()
        with subprocess.Popen(
                ['python', os.path.join(os.getcwd(), 'resources/models/research/object_detection/model_main_tf2.py'),
                 '--pipeline_config_path={path}'.format(path=os.path.join(self.training_dir, 'pipeline.config').replace(os.path.sep, '/')),
                 '--model_dir={path}'.format(path=self.training_dir),
                 '--alsologtostderr'],
                env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr) as out:
            out.communicate()
            print(out.returncode)
        end_time = time.time()

        PrintUtils.info('Training finished! Time taken: {:.2f} minutes'.format((end_time - start_time) / 60))

    def export_model(self):
        """
        Export a model to a zip file
        The user chooses the dir of the model to export
        The user chooses the dir of the exported model

        :return: None
        """
        PrintUtils.inputmsg('Choose the directory of the model to export')
        model_dir = ask_directory(self.root, force=False, initialdir=os.getcwd(), title='Select the directory of the model')
        if model_dir == '':
            return

        PrintUtils.inputmsg('Choose an empty directory for the exported model')
        export_path = ask_empty_directory(self.root, force=False, initialdir=os.getcwd(), title='Select a directory for the exported model')
        if model_dir == '':
            return

        PrintUtils.info('Exporting model')
        proc = subprocess.Popen(
            ['python', os.path.join(os.getcwd(), 'resources/models/research/object_detection/exporter_main_v2.py'),
             '--input_type=image_tensor',
             '--pipeline_config_path={path}'.format(path=os.path.join(model_dir, 'pipeline.config').replace(os.path.sep, '/')),
             '--trained_checkpoint_dir={path}'.format(path=model_dir.replace(os.path.sep, '/')),
             '--output_directory={path}'.format(path=os.path.join(export_path, 'model/').replace(os.path.sep, '/'))],
            env=os.environ.copy())
        proc.wait()

        shutil.copy(self.data_handler.label_map_path, os.path.join(export_path, 'model'))

        PrintUtils.info('Zipping model')
        to_zip(os.path.join(export_path, 'model'), os.path.join(export_path, 'model.zip'))
        PrintUtils.info('Model exported to {}'.format(os.path.join(export_path, 'model.zip')))

    def load_model(self):
        """
        Load an exported model
        The user chooses the exported model zip
        The user chooses a directory to unzip the model to

        :return:
        """
        try:
            """
            saved_model_dir = ask_directory(self.root, initialdir=os.getcwd(), title='Select model directory')

            PATH_TO_LABEL_MAP = os.path.join(saved_model_dir, 'label_map.pbtxt')
            label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1,
                                                                        use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

            self.loaded_model = tf.saved_model.load(os.path.join(saved_model_dir, 'saved_model'))
            """

            PrintUtils.inputmsg('Select the model zip file')
            model_zip = ask_for_file(self.root, force=False, title='Select model zip', initialdir=os.getcwd(),
                                     filetypes=(('ZIP files', '*.zip'),))
            if model_zip is None or model_zip == '':
                return False

            PrintUtils.inputmsg('Select an empty directory for the extracted model')
            extract_dir = ask_empty_directory(self.root, force=False, initialdir=os.getcwd(), title='Select a directory to extract the model')
            if extract_dir == '':
                return False

            extract_zip(model_zip, extract_dir)

            PATH_TO_LABEL_MAP = os.path.join(extract_dir, 'label_map.pbtxt')
            label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1,
                                                                        use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

            self.loaded_model = tf.saved_model.load(os.path.join(extract_dir, 'saved_model'))
            return True
        except Exception as e:
            self.loaded_model = None
            self.category_index = None
            PrintUtils.error('Unable to load the model')
            print(e)
            return False
        # print(model.signatures['serving_default'])
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # output = model.signatures['serving_default'](input_tensor=image_np_expanded)
        # print(output.keys())

    def test_on_image(self, label):
        """
        Predict the current loaded model on a given image and show the prediction on a label
        The user chooses which image to test

        :param label: The label to show the prediction on
        :return: The ImageTk photo
        """
        PrintUtils.inputmsg('Select a jpg image to test on')
        image_file = ask_for_file(self.root, force=False, title='Select test image', initialdir=os.getcwd(),
                                  filetypes=(('JPEG', '*.jpg'),))
        if image_file is None or image_file == '':
            return None

        image_np = load_image_into_numpy_array(image_file)
        image_np = self.__pred_img(image_np)
        img = Image.fromarray(image_np)
        img = ImageOps.contain(img, (800, 800))
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        return photo

    def test_on_video(self):
        """
        Test the current loaded model on a video file
        The user chooses the import video file
        The user chooses the output video file

        :return: None
        """
        PrintUtils.inputmsg('Select a .mp4 video file to test on')
        video_file = ask_for_file(self.root, force=False, title='Select test image', initialdir=os.getcwd(),
                                  filetypes=(('Video File', '*.mp4'),))
        if video_file is None or video_file == '':
            return None

        PrintUtils.inputmsg('Select a .mp4 file name to save the predicted video to')
        out_video_file = ask_save_file(self.root, force=False, title='Select test image', initialdir=os.getcwd(),
                                       filetypes=(('Video File', '*.mp4'),),
                                       defaultextension='*.mp4')
        if out_video_file is None or out_video_file == '':
            return None

        vidcap = cv2.VideoCapture(video_file)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))

        success, frame = vidcap.read()
        count = 0
        writer = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.__pred_img(frame)
            writer.write(image)
            if count % 100 == 0:
                PrintUtils.info(f'Frame {count} out of {length}')
            success, frame = vidcap.read()
            count += 1
        PrintUtils.info('Finished predicting on video')

        vidcap.release()
        writer.release()

    def test_on_live(self, label):
        """
        Test the current loaded model on live camera and show predictions on label

        :param label: The label to show the live predictions on
        :return: None
        """
        vidcap = cv2.VideoCapture(0)
        if vidcap is None or not vidcap.isOpened():
            PrintUtils.info('No camera is available')
            return None

        while label.winfo_exists() == 1:
            success, frame = vidcap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np = self.__pred_img(frame)
            img = Image.fromarray(image_np)
            img = ImageOps.contain(img, (800, 800))
            photo = ImageTk.PhotoImage(img)
            if label.winfo_exists() == 1:
                label.configure(image=photo)

    def __pred_img(self, image_np):
        """
        Predict the input image on the current loaded model and return the predictions visualized on it

        :param image_np: Input image
        :return: The image with predictions visualized
        """
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output = self.loaded_model.signatures['serving_default'](input_tensor=image_np_expanded)
        boxes = output['detection_boxes']
        scores = output['detection_scores']
        classes = output['detection_classes']
        # num_detections = output['num_detections']
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.3,
            # skip_scores=True
        )
        return image_np

    def __prepare_config(self):
        """
        Prepare a config file for testing and save it in the current training directory

        :return: None
        """
        pipeline = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.io.gfile.GFile(os.path.join(os.getcwd(), 'resources/base_pipeline.config'), "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline)

        pipeline.train_config.fine_tune_checkpoint = os.path.join(os.getcwd(),
                                                                  'resources/models/pretrained_model/ssd_resnet50_v1_fpn/checkpoint/ckpt-0').replace(
            os.path.sep, '/')
        pipeline.train_input_reader.label_map_path = self.data_handler.label_map_path.replace(os.path.sep, '/')
        pipeline.train_input_reader.tf_record_input_reader.input_path[0] = self.data_handler.train_record_path.replace(
            os.path.sep, '/')
        pipeline.eval_input_reader[0].label_map_path = self.data_handler.label_map_path.replace(os.path.sep, '/')
        pipeline.eval_input_reader[0].tf_record_input_reader.input_path[
            0] = self.data_handler.eval_record_path.replace(os.path.sep, '/')
        pipeline.eval_config.num_examples = self.data_handler.num_eval

        config_text = text_format.MessageToString(pipeline)
        with tf.io.gfile.GFile(os.path.join(self.training_dir, 'pipeline.config'), "wb") as f:
            f.write(config_text)

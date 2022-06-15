import shutil
from collections import namedtuple

import cv2
from sklearn.model_selection import train_test_split

from ProjectFinal.BasicFunctions import *


class DataHandler:
    def __init__(self, gui):
        self.gui = gui
        self.root = gui.root
        self.extracted_dir = None  # The path to the extracted data dir
        self.split_dir = None  # The path to the split data dir
        self.encoded_dir = None  # The path to the encoded data dir
        self.train_record_path = None  # The path to the train .record file
        self.eval_record_path = None  # The path to the eval .record file
        self.label_map_path = None  # The path to the label_map.pbtxt file
        self.num_train = None  # Number of train input images
        self.num_eval = None  # Number of test input images

    def handle_dataset(self):
        self.__get_data()
        self.__validate_data()
        self.__split_data()
        self.__encode_data()

    def __get_data(self):
        """
        Gets the dataset zip from github or local and extracts it.
        The user chooses to download or use a local zip.
        The user chooses where to extract the zip to.

        :return: None
        """
        data_source = PrintUtils.chooseinputs('Where do you want to get the data from?', ['Github', 'Local'])
        if data_source.lower() == 'github':
            PrintUtils.inputmsg('Choose where do you want the zip to be installed to')
            data_zip = ask_save_file(self.root, force=True, title='Select output zip file', initialdir=os.getcwd(),
                                     filetypes=(('ZIP File', '*.zip'),),
                                     defaultextension='*.zip')
            PrintUtils.info(f'Downloading dataset into {data_zip}...')
            download_file(
                'https://github.com/maayan121/project_python_letters-DL/blob/main/All%20images.zip?raw=true',  # TODO: Change url to dataset link
                data_zip)
            PrintUtils.info('Finished downloading.')
        else:
            PrintUtils.inputmsg('Choose the dataset zip file')
            data_zip = ask_for_file(self.root, force=True, title='Select dataset zip', initialdir=os.getcwd(),
                                    filetypes=(('ZIP files', '*.zip'),))

        PrintUtils.inputmsg('Choose an empty directory where the data will be unzipped to')
        self.extracted_dir = ask_empty_directory(self.root, force=True, initialdir=os.getcwd(), title='Select output directory')
        extract_zip(data_zip, self.extracted_dir)

    def __validate_data(self):
        """
        Check the annotations in the dataset and delete the corrupted files

        :return: None
        """
        PrintUtils.info('Validating dataset...')
        images_dir = os.path.join(self.extracted_dir, 'images')
        anns_dir = os.path.join(self.extracted_dir, 'annotations')
        image_ext = 'jpg'

        cnt = 0
        err_cnt = 0
        error_files = []
        for xml_file in glob(anns_dir + '/*.xml'):
            error = False

            xml_file_name = os.path.split(xml_file)[1]
            try:
                xml_tree = ET.parse(xml_file)
                xml_root = xml_tree.getroot()
            except Exception:
                PrintUtils.error(f'Error parsing file {xml_file_name}')
                error = True

            if not error:
                try:
                    filename = xml_root.find('filename').text
                except Exception:
                    PrintUtils.error(f'Error in file {xml_file_name}, reading filename attribute ')
                    error = True

                try:
                    img_width = int(xml_root.find('size').find('width').text)
                except Exception:
                    PrintUtils.error(f'Error in file {xml_file_name}, reading width attribute')
                    error = True

                try:
                    img_height = int(xml_root.find('size').find('height').text)
                except Exception:
                    PrintUtils.error(f'Error in file {xml_file_name}, reading height attribute')
                    error = True

                try:
                    xml_objects = xml_root.findall('object')
                except Exception:
                    PrintUtils.error(f'Error in file {xml_file_name}, reading objects')
                    error = True

            if not error:
                objects = []
                for i, member in enumerate(xml_objects):
                    obj_err = False

                    try:
                        clazz = member.find('name').text
                    except Exception:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading class name attribute')
                        obj_err = True
                        error = True

                    try:
                        if member.find('bndbox') is None:
                            raise Exception
                    except Exception:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading bndbox attribute')
                        obj_err = True
                        error = True

                    if not obj_err:
                        try:
                            xmin = int(member.find('bndbox').find('xmin').text)
                        except Exception:
                            PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading xmin attribute')
                            obj_err = True
                            error = True
                        try:
                            ymin = int(member.find('bndbox').find('ymin').text)
                        except Exception:
                            PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading ymin attribute')
                            obj_err = True
                            error = True
                        try:
                            xmax = int(member.find('bndbox').find('xmax').text)
                        except Exception:
                            PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading xmax attribute')
                            obj_err = True
                            error = True
                        try:
                            ymax = int(member.find('bndbox').find('ymax').text)
                        except Exception:
                            PrintUtils.error(f'Error in file {xml_file_name}, object {i}, reading ymax attribute')
                            obj_err = True
                            error = True

                    if not obj_err:
                        objects.append((clazz, xmin, ymin, xmax, ymax))

            if not error:
                try:
                    img_path = os.path.join(images_dir, filename + '.' + image_ext)
                    img = cv2.imread(img_path)
                    if img is None:
                        raise Exception
                except Exception:
                    PrintUtils.error(f'Error in file {xml_file_name}, filename attribute ')
                    error = True

            if not error:
                org_height, org_width = img.shape[:2]

                if not org_width == img_width:
                    PrintUtils.error(f'Error in file {xml_file_name}, width {img_width} != {org_width}')
                    error = True
                if not org_height == img_height:
                    PrintUtils.error(f'Error in file {xml_file_name}, width {img_height} != {org_height}')
                    error = True

                for i, (clazz, xmin, ymin, xmax, ymax) in enumerate(objects):
                    if xmin < 0 or xmin > org_width:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, xmin value')
                        error = True
                    if ymin < 0 or ymin > org_height:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, ymin value')
                        error = True
                    if xmax < 0 or xmax > org_width:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, xmax value')
                        error = True
                    if ymax < 0 or ymax > org_height:
                        PrintUtils.error(f'Error in file {xml_file_name}, object {i}, ymax value')
                        error = True

            if error:
                PrintUtils.error(f'Error(s) in file {xml_file_name}')
                error_files.append(xml_file)
                err_cnt += 1
            else:
                cnt += 1

        PrintUtils.info(f'Checked {cnt + err_cnt} files, {cnt} ok, found {err_cnt} errors')
        PrintUtils.info('Deleting files with error: \n{}'.format('\n'.join(error_files)))
        for file in error_files:
            os.remove(file)

    def __split_data(self):
        """
        Split the data to train and validation
        The user chooses the directory for the split data

        :return: None
        """
        PrintUtils.inputmsg('Select an empty folder for the split dataset')
        self.split_dir = ask_empty_directory(self.root, force=True, initialdir=os.getcwd(), title='Select split data directory')

        anns_dir = os.path.join(self.extracted_dir, 'annotations')

        train_dir = os.path.join(self.split_dir, 'training/')
        eval_dir = os.path.join(self.split_dir, 'evaluating/')

        os.mkdir(train_dir)
        os.mkdir(eval_dir)

        files = glob(anns_dir + '/*.xml')
        train, eval = train_test_split(files, train_size=0.8, test_size=0.2, shuffle=True, random_state=42)
        PrintUtils.info(f'Train size: {len(train)}')
        PrintUtils.info(f'Eval size: {len(eval)}')

        PrintUtils.info(f'Copying training files to {train_dir}')
        for file in train:
            shutil.copy(file, train_dir)

        PrintUtils.info(f'Copying eval files to {eval_dir}')
        for file in eval:
            shutil.copy(file, eval_dir)

    def __encode_data(self):
        """
        Encode the split labels to csv files
        then encode to tensorflow record files.
        Creates the label_map.pbtxt

        The user chooses the directory for the encoded files
        :return: None
        """
        PrintUtils.inputmsg('Select an empty folder for the encoded dataset')
        self.encoded_dir = ask_empty_directory(self.root, force=True, initialdir=os.getcwd(), title='Select encoded directory')

        train_dir = os.path.join(self.split_dir, 'training/')
        eval_dir = os.path.join(self.split_dir, 'evaluating/')

        train_df, classes_train = xml_to_csv(train_dir, ['pistol'])
        eval_df, classes_eval = xml_to_csv(eval_dir, ['pistol'])

        # Convert train and eval data to csv
        PrintUtils.info('Converting train labels to csv...')
        train_df.to_csv(os.path.join(self.encoded_dir, 'train_labels.csv'), index=None)
        PrintUtils.info('Converting eval labels to csv...')
        eval_df.to_csv(os.path.join(self.encoded_dir, 'eval_labels.csv'), index=None)

        # Create the label_map.pbtxt
        PrintUtils.info('Creating label map file...')
        classes = list(set(classes_train + classes_eval))
        PrintUtils.info('Classes found: [{}]'.format(', '.join(classes)))
        pbtxt_content, label_to_id = create_pbtxt(classes)
        self.label_map_path = os.path.join(self.encoded_dir, 'label_map.pbtxt')
        with open(self.label_map_path, 'w') as f:
            f.write(pbtxt_content)

        # Create the record files
        data_f = namedtuple('data', ['filename', 'object'])
        self.train_record_path = os.path.join(self.encoded_dir, 'train_labels.record')
        self.eval_record_path = os.path.join(self.encoded_dir, 'eval_labels.record')

        # Create the train record files
        PrintUtils.info('Creating train record files...')
        gb = train_df.groupby('filename')
        data = [data_f(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        self.num_train = 0
        with tf.io.TFRecordWriter(self.train_record_path) as writer:
            for d in data:
                tf_example = create_tf_example(d, os.path.join(self.extracted_dir, 'images'), label_to_id)
                writer.write(tf_example.SerializeToString())
                self.num_train += 1

        # Create the eval record files
        PrintUtils.info('Creating eval record files...')
        gb = eval_df.groupby('filename')
        data = [data_f(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        self.num_eval = 0
        with tf.io.TFRecordWriter(self.eval_record_path) as writer:
            for d in data:
                tf_example = create_tf_example(d, os.path.join(self.extracted_dir, 'images'), label_to_id)
                writer.write(tf_example.SerializeToString())
                self.num_eval += 1

import os
import cv2
from src.utils.imgprehandle import blur
import xlwt
import torch
from src.musiq import Musiq

img_definition_func_name_list = ["musiq"]
blur_type_name_list = ["resize"]
blur_degree_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
img_root = "/workspace/huangniu_demo/image_definition_det/test_data_with_mask"  # could be changed
all_paths = [os.path.join(img_root, img_name) for img_name in os.listdir(img_root)]

row_dict = {}
row_num = 1
for blur_type_name in blur_type_name_list:
    for blur_degree in blur_degree_list:
        row_key = blur_type_name + str(blur_degree)
        row_dict[row_key] = row_num
        row_num += 1

column_dict = {}
for n, path in enumerate(all_paths):
    column_key = path.split("/")[-1].split(".")[0]
    column_dict[column_key] = 1 + 2 * n


def get_test_imgs():
    test_imgs = []
    test_imgs_params = []
    for path in all_paths:
        img_bgr = cv2.imread(path)
        for blur_type_name in blur_type_name_list:
            for blur_degree in blur_degree_list:
                blured_bgr = blur(img_bgr,
                                  blur_type=blur_type_name,
                                  blur_degree=blur_degree)
                test_imgs.append(blured_bgr[:, :, ::-1])
                test_imgs_params.append([blur_type_name, blur_degree, path.split("/")[-1].split(".")[0]])
    return test_imgs, test_imgs_params


def cal_definition(definition_func_name, imgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = Musiq(musiq_config="/workspace/huangniu_demo/image_definition_det/src/configs/musiq.yaml",
              musiq_weights="/workspace/huangniu_demo/image_definition_det/src/weights/musiq_paq2piq_ckpt-364c0c84_2.pth",
              device=device)
    res = t.batch_inference(imgs)
    print(res)
    return res


def record_res(res, test_imgs, test_imgs_params, out_img_root="", definition_func_name=""):
    book = xlwt.Workbook()
    sheet = book.add_sheet(img_root.split("/")[-1])

    for k, v in row_dict.items():
        sheet.write(v, 0, k)
    for k, v in column_dict.items():
        sheet.write(0, v, k)
        sheet.write(0, v + 1, "definition")

    # record res
    for res_i, definition in enumerate(res):
        test_img = test_imgs[res_i]
        test_imgs_param = test_imgs_params[res_i]
        img_row = row_dict[test_imgs_param[0] + str(test_imgs_param[1])]
        img_column = column_dict[test_imgs_param[2]]
        definition_row = img_row
        definition_column = img_column + 1

        # write img
        blured_img_name = "{}_{}_{}.bmp".format(
            test_imgs_param[2],
            test_imgs_param[0],
            str(test_imgs_param[1]))
        blured_img_path = os.path.join(out_img_root, blured_img_name)
        cv2.imwrite(blured_img_path, cv2.resize(test_img, (160, 160))[:, :, ::-1])
        sheet.insert_bitmap(blured_img_path, img_row, img_column)

        # write definition
        sheet.write(definition_row, definition_column, str(float(definition)))
    book.save('{}_{}.xls'.format(definition_func_name, img_root.split("/")[-1]))


def analyze(res):
    pass


if __name__ == "__main__":
    definition_func_name = "musiq"
    test_imgs, test_imgs_params = get_test_imgs()
    res = cal_definition(definition_func_name, test_imgs)
    record_res(res, test_imgs, test_imgs_params,
               out_img_root="/workspace/huangniu_demo/image_definition_det/test_data_blured",
               definition_func_name=definition_func_name)

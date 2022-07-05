import os
import re
import logging
from sys import path
from spacy.util import set_data_path

import spacy
from text_cls_tool import text_classify

ner_dir = os.path.split(os.path.realpath(__file__))[0]
path.append(ner_dir)
import format_output
from ner_utils import *
from format4NER import format_4NER_bill
from entityParse import *
from ner_post_handle import *
from parse_region import *

global nlp_bjqd
global nlp_bjjsd
global nlp_qgtyqd
global nlp_qgtyjsd


def init_module():
    global nlp_bjqd, nlp_bjjsd, nlp_qgtyqd, nlp_qgtyjsd, model_dict, type_dict
    # spacy2.1.4训练的模型
    # model_bjqd = "infoParse/models/ner_bj_qd"
    # model_bjjsd = "infoParse/models/ner_bj_jsd"
    # model_qgtyqd = "infoParse/models/qd_add_shandong_v1"
    # model_qgtyjsd = "infoParse/models/jsd_add_shandong_v2"

    # spacy2.3.4训练的模型
    model_bjqd = os.path.join(ner_dir, "models/ner_bjqd")
    model_bjjsd = os.path.join(ner_dir, "models/ner_bjjsd")
    model_qgtyqd = os.path.join(ner_dir, "models/ner_qgtyqd")
    model_qgtyjsd = os.path.join(ner_dir, "models/ner_qgtyjsd")
    set_data_path(model_bjqd)

    nlp_bjqd = spacy.load(model_bjqd)
    nlp_bjjsd = spacy.load(model_bjjsd)
    nlp_qgtyqd = spacy.load(model_qgtyqd)
    nlp_qgtyjsd = spacy.load(model_qgtyjsd)

    model_dict = {"北京清单": 'nlp_bjqd', "北京结算单": 'nlp_bjjsd', "全国统一清单": 'nlp_qgtyqd', "全国统一结算单": 'nlp_qgtyjsd'}
    type_dict = {"北京结算单": 10, "全国统一结算单": 11, "北京清单": 11, '全国统一清单': 12}


def merge_filter(pos_list, txt_list, char_score_list):
    pos_list, txt_list, char_score_list = removeEmpty(pos_list, txt_list, char_score_list)
    output_y, output_x, output_txt, output_s = line_split(char_score_list, txt_list, pos_list, False)
    content = " ".join([" ".join(item) for item in output_txt])
    if '北京协和医院门诊费用清单' in content:
        return False
    elif '姓名：马俊良 收据号：20220124981011' in content:
        return False
    else:
        return True


def info_parse(pos_list, txt_list, char_score_list, textType):
    bj_map = {"清单": "北京清单", "结算单": "北京结算单"}
    qg_map = {"清单": "全国统一清单", "结算单": "全国统一结算单"}
    pred_cls = ''
    pos_list, txt_list, char_score_list = removeEmpty(pos_list, txt_list, char_score_list)
    output_y, output_x, output_txt, output_s = line_split(char_score_list, txt_list, pos_list, False)
    len_line = [len(line) for line in output_txt]
    content = " ".join([" ".join(item) for item in output_txt])

    txt_scores = cal_text_score(output_s, content)
    assert len(txt_scores) == len(content)
    content = handle_content(content)
    if textType == "北京":
        try:
            pred_cls = text_classify.predictLabel(content)
            textType = bj_map[pred_cls]
        except Exception as e:
            logging.warning(e)
            textType = '北京清单'
    elif textType == "全国":
        try:
            pred_cls = text_classify.predictLabel(content)
            textType = qg_map[pred_cls]
        except Exception as e:
            logging.warning(e)
            textType = '全国统一清单'

    ner_kvlist = list()
    ent_infolist = list()
    if textType == '北京结算单' or textType == '全国统一结算单':
        content = content.replace('，', ',')
    result = Score(eval(model_dict[textType]), content)
    for ent in result:
        [start, end, text, label, score] = ent
        row_ind, col_s_ind, col_e_ind = get_ent_ind(content, start, end, len_line)
        xs = output_x[row_ind][col_s_ind]
        ys = output_y[row_ind][col_s_ind]
        txt_score = txt_scores[start:end]
        txt_score = list(filter(lambda x: x != 0, txt_score))
        try:
            txt_score_av = sum(txt_score) / len(txt_score)
        except:
            txt_score_av = 0
        ent_infolist.append([xs, ys, text, label, (row_ind, col_s_ind, col_e_ind), (score, txt_score_av)])
        ner_kvlist.append((text, label, (row_ind, col_s_ind), (score, txt_score_av)))

    #     logging.debug("{} | {} | {} | {} | {} |{}".format(label, text, (row_ind, col_s_ind), score, txt_score_av, txt_score))

    return ner_kvlist, ent_infolist, content, output_x, output_y, output_txt, output_s, textType, pred_cls


def fromat_bjjsd(ner_kvlist, output_x, output_y, output_txt, textType, backgroundInfo={}):
    type_code = type_dict[textType]
    res = format_4NER_bill(ner_kvlist, (output_x, output_y, output_txt), type_code, backgroundInfo)

    return res


def fromat_qgtyjsd(ner_kvlist, output_x, output_y, output_txt, textType, backgroundInfo={}):
    type_code = type_dict[textType]
    res = format_4NER_bill(ner_kvlist, (output_x, output_y, output_txt), type_code, backgroundInfo)
    try:
        res = update_fields(res, ner_kvlist, output_x, output_y)
    except:
        pass

    return res


def format_bjqd(ner_kvlist, output_x, output_y, output_txt, output_s):
    parser = ParseDetailBj(ner_kvlist, output_x, output_y, output_txt, output_s, config_dict)
    ner_result = parser.run()
    res = format_output.gen_format_output(ner_result)

    return res


def format_qgtyqd(ner_kvlist, output_x, output_y, output_txt, output_s, province_post, city_post, params):
    parser = ParseDetailComm(ner_kvlist, output_x, output_y, output_txt, output_s, config_dict)
    res, other_ents, amount_ents, need_handle = parser.run()
    medical_level_map2(res)
    region = ParseRegion(res, province_post, city_post)
    province, city, hname, hospital_name = region.call_province_city()
    logging.debug('最终提取的省份和城市: {}, {}'.format(province, city))
    logging.debug('第一阶段输出: {}'.format(res))
    logging.debug('params: {}'.format(params))
    cities = [city]
    try:
        res = ner_result_check(res, need_handle)
    except:
        logging.debug("ner_result_check 未生效")
        pass
    coord_info = (output_x, output_y, output_txt)
    res = format_output.gen_format_output_v1(res, other_ents, amount_ents, coord_info, province, city, params)

    return res, province, city, hname, hospital_name

from tqdm.auto import tqdm
import pandas as pd
import re
import sys

import sys
import time
f=open("myprint.txt","w+")
sys.stdout=f

tagset={'VA','VC','VE','VV','NR','NT','NN','LC','PN','DT','CD','OD','M','AD','P','CC',
        'CS','DEC','DEG','DER','DEV','SP','AS','ETC','SP','MSP','IJ','ON','PU','JJ','FW','LB','SB','BA','NOI'}

path='ctbpred.csv'
data=pd.read_csv(path)


def illegal_check(l):
    for i in l:
        if i not in tagset:
            print('ILLEGAL!!:',i)
            return True

    return False

def division_check(l):
    pattern=r'\d\/\d'
    new = re.sub(pattern,'几分之几',l)
    return new


result = {'fully correct':[],  #3287
          'predict less but correct':[],  # 31
          'segmented wrong':[],  #87
          'tagged wrong':[],  # 1288
          'illegal tag':[],  # 0  事实上不存在，NOI tag表示噪音，旧文档没记录
          'seg&tagged wrong':[],  # 69
          'all wrong':[],  # 3345
          'weird':[],  # 27    # 换个分隔符可能算起来正确率还高些
          }
# 短句预测全对的更多
# 预测多了少了
#max length限制
#分词错误和标注错误分开统计
#预测不存在tag的


for i,row in tqdm(data.iterrows()):
    score, ref, pred = row[0], row[1], row[2]
    ref = division_check(ref)
    pred = division_check(pred)

    if score == 1.0:
        result['fully correct'].append([row[1],row[2]])
    else:
        if len(ref) == len(pred):

            try:
                ref = ref.strip().split('/')
                pred = pred.strip().split('/')
                r_words = [i.split('_')[0] for i in ref]
                r_tags = [i.split('_')[1] for i in ref]  # 比如 ‘需要_VV/的_DEC/1/3_CD/票数_NN’
                p_words = [i.split('_')[0] for i in pred]
                p_tags = [i.split('_')[1] for i in pred]

                if r_words != p_words:
                    result['segmented wrong'].append([row[1],row[2]])
                    if r_tags != p_tags:
                        result['seg&tagged wrong'].append([row[1],row[2]])
                elif r_tags != p_tags:
                    result['tagged wrong'].append([row[1],row[2]])
                    if illegal_check(r_tags):
                        result['illegal tag'].append([row[1],row[2]])
            except IndexError:
                result['weird'].append([row[1],row[2]])

        else:
            try:
                ref = ref.strip().split('/')
                pred = pred.strip().split('/')
                ref.pop()
                pred.pop()
                r_words = [i.split('_')[0] for i in ref]
                r_tags = [i.split('_')[1] for i in ref]
                p_words = [i.split('_')[0] for i in pred]   # 这不align怎么搞得清？ 185行 ',_PU/2/3_CD/以上_LC/'=>',_PU/2_CD//_PU/2_CD/以上_LC'
                p_tags = [i.split('_')[1] for i in pred]

                if p_tags == r_tags[:len(p_tags) + 1]:
                    result['predict less but correct'].append([row[1],row[2]])
                else:
                    result['all wrong'].append([row[1],row[2]])

            except IndexError:
                result['weird'].append([row[1],row[2]])



#from IPython import embed; embed()
print('irregular format')
for i in result['weird']:
    print(i)
    print('                 ')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('tagged wrong')
for i in result['tagged wrong']:
    print(i)
    print('               ')
print('-----------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('segmented wrong')
for i in result['segmented wrong']:
    print(i)
    print('            ')
print('-----------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('seg adn tag wrong')
for i in result['seg&tagged wrong']:
    print(i)
    print('              ')
print('-----------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('predict less but right')
for i in result['predict less but correct']:
    print(i)
    print('              ')
print('-----------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------')
print('all wrong')
for i in result['all wrong']:
    print(i)
    print('            ')


'''r,p
"中央台_NR/记者_NN/李涛_NR/报道_VV/:_PU/全_DT/国_NN/人大_NN/常委会_NN/《_PU/刑事_NN/诉讼法_NN/》_PU/执法_JJ/检查组_NN/昨天_NT/举行_VV/全体_JJ/会议_NN/,_PU/总结_VV/今年_NT/9月_NT/对_P/天津_NR/等_ETC/6_CD/个_M/省_NN/、_PU/自治区_NN/、_PU/直辖市_NN/贯彻_VV/实施_VV/《_PU/刑事_NN/诉讼法_NN/》_PU/的_DEG/检查_NN/情况_NN/。_PU ",
"中央台_NR/记者_NN/李涛_NR/报道_VV/:_PU/全_DT/国_NN/人大_NN/常委会_NN/《_PU/刑事_NN/诉讼法_NN/》_PU/执法_NN/检查组_NN/昨天_NT/举行_VV/全体_DT/会议_NN/,_PU/总结_VV/今年_NT/9月_NT/对_P/天津_NR/等_ETC/6_CD/个_M/省_NN/、_PU/自治区_NN/、_PU/直辖市_NN/"
"中共_NR/中央_NN/政治局_NN/常委_NN/、_PU/全_DT/国_NN/人大_NN/常委会_NN/委员长_NN/李鹏_NR/在_P/会_NN/上_LC/指出_VV/:_PU/各_DT/级_M/公安_NN/、_PU/司法_NN/机关_NN/要_VV/采取_VV/切实_AD/有效_VA/的_DEC/措施_NN/,_PU/努力_AD/解决_VV/《_PU/刑诉法_NN/》_PU/贯彻_VV/实施_VV/中_LC/人民_NN/群众_NN/反映_VV/比较_AD/强烈_VA/的_DEC/问题_NN/,_PU/维护_VV/国家_NN/根本_JJ"
"中共_NR/中央_NN/政治局_NN/常委_NN/、_PU/全_DT/国_NN/人大_NN/常委会_NN/委员长_NN/李鹏_NR/在_P/会_NN/上_LC/指出_VV/:_PU/各_DT/级_M/公安_NN/、_PU/司法_NN/机关_NN/要_VV/采取_VV/切实_VA/有效_VA/的_DEC/措施_NN/,_PU/努力_AD/解决_VV/《_PU/刑诉法_NN"
"李鹏_NR/强调_VV/:_PU/贯彻_VV/依法_AD/治国_VV/方略_NN/是_VC/坚持_VV/人民_NN/民主_NN/专政_NN/的_DEC/重要_JJ/保证_NN/,_PU/就_AD/这_AD/要求_VV/公安_NN/、_PU/司法_NN/机关_NN/在_P/执法_JJ/过程_NN/中_LC/必须_VV/依法_AD/办事_VV/。_PU "
"李鹏_NR/强调_VV/:_PU/贯彻_VV/依法_AD/治国_VV/方略_NN/是_VC/坚持_VV/人民_NN/民主_NN/专政_NN/的_DEC/重要_JJ/保证_NN/,_PU/就_P/这_PN/要求_VV/公安_NN/、_PU/司法_NN/机关_NN/在_P/执法_NN/过程_NN/中_LC/必须_VV/依法_AD/办事_VV/。_PU "
"从_P/这_DT/次_M/执法_JJ/检查_NN/情况_NN/看_VV/,_PU/在_P/贯彻_VV/实施_VV/《_PU/刑诉法_NN/》_PU/中_LC/也_AD/暴露_VV/出_VV/一些_CD/问题_NN/,_PU/尤其是_AD/超期_AD/积压_VV/、_PU/刑讯_VV/逼供_VV/、_PU/保障_VV/律师_NN/依法_AD/履行_VV/职务_NN/等_ETC/方面_NN/的_DEG/问题_NN/。_PU "
"从_P/这_DT/次_M/执法_NN/检查_NN/情况_NN/看_VV/,_PU/在_P/贯彻_VV/实施_VV/《_PU/刑诉法_NN/》_PU/中_LC/也_AD/暴露_VV/出_VV/一些_CD/问题_NN/,_PU/尤其是_AD/超期_AD/积压_VV/、_PU/刑讯_NN/逼供_VV/、_PU/保障_VV/律师_NN/依法_AD/履行_VV/职务_NN/"
"这些_DT/问题_NN/如果_CS/长期_AD/得_VV/不_AD/到_VV/解决_NN/,_PU/将_AD/影响_VV/公安_NN/、_PU/司法_NN/机关_NN/在_P/人民_NN/心目_NN/中_LC/的_DEG/形象_NN/,_PU/影响_VV/法律_NN/的_DEG/权威性_NN/,_PU/影响_VV/社会主义_NN/民主_NN/与_CC/法制_NN/建设_NN/的_DEG/大局_NN/。_PU "
"这些_DT/问题_NN/如果_CS/长期_AD/得_VV/不_AD/到_VV/解决_NN/,_PU/将_AD/影响_VV/公安_NN/、_PU/司法_NN/机关_NN/在_P/人民_NN/心目_NN/中_LC/的_DEG/形象_NN/,_PU/影响_VV/法律_NN/的_DEG/权威性_NN/,_PU/影响_VV/社会主义_NN/民主_NN/与_CC/法制_NN/建设_NN/的_DEG/"
"'''
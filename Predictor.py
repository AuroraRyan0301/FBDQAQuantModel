import os
from .model import *
import numpy as np

class Predictor():
    def __init__(self):
        self.model = Res50Transformer(num_classes = 3)   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dicts = torch.load("./mmpc/Res50Transformer_80000.pth",map_location=self.device) ## 注意这里要确保路径能被加载, 使用./mmpc/[my_model_file]路径
        self.model.load_state_dict(state_dicts)  ## 加载模型参数，此处应和训练时的保存方式一致
        self.model.to(self.device)
        self.model.eval()

    def predict(self,x):
        with torch.no_grad():
            x = self.preprocess(x).to(self.device) # 移动到GPU
            y = self.model(x)                      # shape:[1, output]
            y = y.cpu().numpy()                    # 别忘了移动回CPU以便后续处理
            y = self.generate_signal(y, single_label=True) # shape:[1, label_num]
            y = list(y.reshape(-1))                # shape:[label_num]
            return y
    
    ## 这里不重要，预处理方式因人而异
    def preprocess(self, df):
        ''' 数据预处理 '''        
        df['bid1'] = df['bid1']+1
        df['bid2'] = df['bid2']+1
        df['bid3'] = df['bid3']+1
        df['bid4'] = df['bid4']+1
        df['bid5'] = df['bid5']+1
        df['ask1'] = df['ask1']+1
        df['ask2'] = df['ask2']+1
        df['ask3'] = df['ask3']+1
        df['ask4'] = df['ask4']+1
        df['ask5'] = df['ask5']+1

        # 量价组合
        df['spread1'] =  df['ask1'] - df['bid1']
        df['spread2'] =  df['ask2'] - df['bid2']
        df['spread3'] =  df['ask3'] - df['bid3']
        df['mid_price1'] =  df['ask1'] + df['bid1']
        df['mid_price2'] =  df['ask2'] + df['bid2']
        df['mid_price3'] =  df['ask3'] + df['bid3']
        df['weighted_ab1'] = (df['ask1'] * df['bsize1'] + df['bid1'] * df['asize1']) / (df['bsize1'] + df['asize1'])
        df['weighted_ab2'] = (df['ask2'] * df['bsize2'] + df['bid2'] * df['asize2']) / (df['bsize2'] + df['asize2'])
        df['weighted_ab3'] = (df['ask3'] * df['bsize3'] + df['bid3'] * df['asize3']) / (df['bsize3'] + df['asize3'])

        df['vol1_rel_diff']   = (df['bsize1'] - df['asize1']) / (df['bsize1'] + df['asize1'])
        df['volall_rel_diff'] = (df['bsize1'] + df['bsize2'] + df['bsize3'] + df['bsize4'] + df['bsize5'] \
                        - df['asize1'] - df['asize2'] - df['asize3'] - df['asize4'] - df['asize5'] ) / \
                        ( df['bsize1'] + df['bsize2'] + df['bsize3'] + df['bsize4'] + df['bsize5'] \
                        + df['asize1'] + df['asize2'] + df['asize3'] + df['asize4'] + df['asize5'] )

        df['amount'] = df['amount_delta'].map(np.log1p)

        df['bid1'] = df['bid1']-1
        df['bid2'] = df['bid2']-1
        df['bid3'] = df['bid3']-1
        df['bid4'] = df['bid4']-1
        df['bid5'] = df['bid5']-1
        df['ask1'] = df['ask1']-1
        df['ask2'] = df['ask2']-1
        df['ask3'] = df['ask3']-1
        df['ask4'] = df['ask4']-1
        df['ask5'] = df['ask5']-1

        feature_col_names = ['bid1','bsize1',
                     'bid2','bsize2',
                     'bid3','bsize3',
                     'bid4','bsize4',
                     'bid5','bsize5',
                     'ask1','asize1',
                     'ask2','asize2',
                     'ask3','asize3',
                     'ask4','asize4',
                     'ask5','asize5',
                     'spread1','mid_price1',
                     'spread2','mid_price2',
                     'spread3','mid_price3',
                     'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                     'vol1_rel_diff','volall_rel_diff'
                    ]
        data = torch.tensor(df[feature_col_names].values).to(torch.float32).unsqueeze(0).unsqueeze(0)
        return data
    
    def generate_signal(self, predict_matrix, class_num=3, single_label=True):
        '''
        Args:
            predict_matrix: np [sample_num, class_num * label_num] if single_label = False
                            np [sample_num, class_num] if single_label = True
        Returns:
            signal: np [sample_num, label_num] if single_label = False
                    np [sample_num] if single_label = True
        '''
        if single_label:
            signal = predict_matrix.argmax(axis=1) # shape:[sample_num]
            return signal
        else:
            signal =  predict_matrix.reshape(predict_matrix.shape[0], class_num, predict_matrix.shape[1] // class_num)
            signal = signal.transpose(0,2,1) # shape:[sample_num, label_num, class_num]
            signal = signal.argmax(axis=2) # shape:[sample_num, label_num]
            return signal


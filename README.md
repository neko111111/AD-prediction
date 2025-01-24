# AD-prediction

本模型用于阿尔兹海默症的多组学预测，具体使用到了多头跨自注意力模型、自动编码器和注意力融合三个模块，在二分类预测问题上取得了不错的效果。 
AE.py:自动编码器模块 
Self_Attention.py:自注意力模块 
attention_fusion.py:注意力融合模块 
feat_importance.py:标志物挖掘 
main.py:分类模型运行 
main_biomarker.py:标志物挖掘运行 
model_init.py:模型组合 
train_test.py:分类模型细节 
utils.py:工具 

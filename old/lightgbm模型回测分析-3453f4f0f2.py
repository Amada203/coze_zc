# 导入必要的库
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# 读取预测结果文件
try:
    with open('/Users/apple/Downloads/lightgbm价格预测结果.json', 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(prediction_data)
    
    # 检查数据格式
    if 'y_true' not in df.columns or 'y_pred' not in df.columns:
        raise ValueError("预测结果文件中缺少必要的列'y_true'或'y_pred'")
    
    # 计算评估指标
    accuracy = accuracy_score(df['y_true'], df['y_pred'])
    report = classification_report(df['y_true'], df['y_pred'], output_dict=True)
    conf_matrix = confusion_matrix(df['y_true'], df['y_pred'])
    roc_auc = roc_auc_score(df['y_true'], df['y_pred'])
    
    # 准备评估结果文本
    result_text = "LightGBM模型回测评估结果\n"
    result_text += "=========================\n\n"
    result_text += f"模型准确率: {accuracy:.4f}\n"
    result_text += f"ROC AUC分数: {roc_auc:.4f}\n\n"
    
    result_text += "分类报告:\n"
    result_text += f"类别0的精确率: {report['0']['precision']:.4f}\n"
    result_text += f"类别0的召回率: {report['0']['recall']:.4f}\n"
    result_text += f"类别0的F1分数: {report['0']['f1-score']:.4f}\n\n"
    result_text += f"类别1的精确率: {report['1']['precision']:.4f}\n"
    result_text += f"类别1的召回率: {report['1']['recall']:.4f}\n"
    result_text += f"类别1的F1分数: {report['1']['f1-score']:.4f}\n\n"
    result_text += f"宏平均F1分数: {report['macro avg']['f1-score']:.4f}\n"
    result_text += f"加权平均F1分数: {report['weighted avg']['f1-score']:.4f}\n\n"
    
    result_text += "混淆矩阵:\n"
    result_text += f"真负例(TN): {conf_matrix[0][0]}\n"
    result_text += f"假正例(FP): {conf_matrix[0][1]}\n"
    result_text += f"假负例(FN): {conf_matrix[1][0]}\n"
    result_text += f"真正例(TP): {conf_matrix[1][1]}\n\n"
    
    result_text += "模型优化建议:\n"
    if report['0']['f1-score'] < 0.8:
        result_text += "- 类别0的预测性能较差，建议增加类别0的样本权重或使用过采样技术\n"
    if report['1']['f1-score'] < 0.9:
        result_text += "- 类别1的预测性能有提升空间，可以尝试调整模型超参数\n"
    if conf_matrix[0][1] > conf_matrix[0][0] * 0.3:
        result_text += "- 假正例(FP)比例较高，建议提高分类阈值或增加惩罚项\n"
    if conf_matrix[1][0] > conf_matrix[1][1] * 0.2:
        result_text += "- 假负例(FN)比例较高，建议降低分类阈值或增加召回率优化\n"
    
    # 保存评估结果
    output_file = 'lightgbm模型回测评估结果.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    print(f"评估结果已保存到文件: {output_file}")
    
except FileNotFoundError:
    print("错误：未找到预测结果文件 '/Users/apple/Downloads/lightgbm价格预测结果.json'")
except json.JSONDecodeError:
    print("错误：预测结果文件格式不正确，无法解析JSON")
except Exception as e:
    print(f"发生未知错误: {str(e)}")
# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 设置页面标题和布局
st.set_page_config(page_title="心血管代谢性共病医疗预测模型", layout="wide", page_icon="🏥")

# 模型文件路径
MODEL_PATH = "sklearn_LightGBM_best_model.sav"

# === 连续变量的归一化参数 ===
feature_mins = np.array([
    65.0,      # age_min
    35.2,      # temperature_min  
    36.0,      # pulse_min
    78.0,      # systolic_min
    50.0,      # diastolic_min
    0.1,       # left_naked_eye_min
    0.1,       # right_naked_eye_min
    31.0,      # weight_kg_min
    36.0,      # heart_rate_min
    50.0,      # HGB_min
    1.0,       # WBC_min
    31.0,      # PLT_min
    1.1,       # glucose_min
    22.0,      # Creatinine_min
    1.0,       # Urea_min
    0.88,      # TC_min
    0.3,       # TG_min
    0.1,       # LDL_C_min
    0.1326,    # HDL_C_min
    -0.2857    # RFM_min
])

feature_maxs = np.array([
    112.0,     # age_max
    38.0,      # temperature_max
    116.0,     # pulse_max
    192.0,     # systolic_max
    133.0,     # diastolic_max
    2.0,       # left_naked_eye_max
    2.0,       # right_naked_eye_max
    180.0,     # weight_kg_max
    130.0,     # heart_rate_max
    234.0,     # HGB_max
    28.86,     # WBC_max
    906.0,     # PLT_max
    22.9,      # glucose_max
    761.0,     # Creatinine_max
    41.3,      # Urea_max
    13.82,     # TC_max
    11.9,      # TG_max
    8.19,      # LDL_C_max
    5.0,       # HDL_C_max
    58.0225    # RFM_max
])

# 特征名称映射（20个连续变量 + 7个分类变量）
continuous_feature_names = [
    'age', 'temperature', 'pulse', 'systolic', 'diastolic',
    'left_naked_eye', 'right_naked_eye', 'weight_kg', 'heart_rate',
    'HGB', 'WBC', 'PLT', 'glucose', 'Creatinine', 'Urea',
    'TC', 'TG', 'LDL_C', 'HDL_C', 'RFM'
]

categorical_feature_names = [
    'gender', 'exercise_freq', 'smoke', 'heart_rhythm', 
    'Bscan', 'count', 'tcm_pinghe'
]

all_feature_names = continuous_feature_names + categorical_feature_names

@st.cache_resource
def load_model():
    """
    加载LGBM模型
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            content = pickle.load(file)
            
        # 检查加载的对象类型
        print(f"加载的对象类型: {type(content)}")
        
        # 如果是字典，提取模型
        if isinstance(content, dict) and 'model' in content:
            model = content['model']
                           
            return model, content
        else:
                       
            # 创建包含模型信息的字典
            model_content = {
                'model': content,
                'best_threshold': 0.5,  # 默认阈值
                'train_sensitivity': 0.0  # 默认敏感度
            }
            
            return content, model_content
            
    except FileNotFoundError:
        st.error(f"未找到模型文件: {MODEL_PATH}")
        return None, None
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        # 显示详细错误信息
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return None, None

# 初始化session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# 加载模型
model, model_content = load_model()

# 移除侧边栏的阈值设置，直接在内置代码中设置
custom_threshold = 0.3  # 直接设置为固定值，不展示给用户

# 欢迎页面
def show_welcome_page():
    st.title("🏥 欢迎使用心血管代谢性共病风险评估预测系统MyCMMrisk")
    
    st.markdown("""
    ## 📋 系统介绍
    
    **MyCMMrisk** 是一个基于人工智能的心血管代谢性共病风险评估工具，旨在帮助医疗工作者：
    
    - 🔍 **早期识别** 心血管代谢疾病高风险人群
    - 📊 **科学评估** 个体患病风险概率
    - 💡 **辅助决策** 为临床干预提供参考依据
    - 🎯 **精准预防** 实现个性化健康管理
    
    ## 🎯 适用人群
    
    本系统适用于65岁及以上中老年人群，特别关注：
    - 患有高血压、糖尿病、血脂异常等基础疾病的人群
    - 有心血管代谢疾病家族史的人群
    - 希望了解自身健康风险状况的人群
    
    ## 📝 评估流程
    
    评估过程分为四个步骤，请按顺序填写相关信息：
    
    1. **👤 基本信息** - 年龄、性别、身体测量指标等
    2. **🔬 生理指标** - 体温、血压、视力、心率等
    3. **💉 血液生化指标** - 血常规、血脂、血糖等实验室指标
    4. **👤 生活习惯** - 生活方式、中医体质等信息
    
    ## ⚠️ 重要说明
    
    - 本系统预测结果仅供参考，不能替代专业医疗诊断
    - 请确保输入数据的准确性和完整性
    - 如有疑问，请咨询专业医疗人员
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 开始", type="primary", use_container_width=True):
            st.session_state.current_page = 'basic_info'
            st.rerun()

# 基本信息页面
def show_basic_info_page():
    st.header("👤 基本信息")
    st.info("请填写患者的基本信息和身体测量指标")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("年龄", min_value=65, max_value=112, value=70, key="age")
        gender = st.radio("性别", [1, 2], format_func=lambda x: "男" if x == 1 else "女", horizontal=True, key="gender")
        count = st.selectbox("高血压、糖尿病、血脂异常中，您患有几个疾病", options=[1, 2, 3], format_func=lambda x: f"{x}个", index=0, key="count")

    with col2:
        height_cm = st.number_input("身高 (cm)", min_value=80.0, max_value=250.0, value=165.0, step=0.1, key="height_cm")
        WC_cm = st.number_input("腰围 (cm)", min_value=40.0, max_value=200.0, value=80.0, step=0.1, key="WC_cm")
        weight_kg = st.number_input("体重 (kg)", min_value=31.0, max_value=180.0, value=65.0, step=0.1, key="weight_kg")
        
        # 自动计算RFM
        if height_cm > 0 and WC_cm > 0:
            # 根据性别设置A值：男性A=0，女性A=1
            A = 1 if gender == 2 else 0  # 1=男，2=女
            RFM = 64 - (20 * height_cm / WC_cm) + (12 * A)
            st.metric("计算得到的相对脂肪质量 (RFM)", f"{RFM:.2f}")
            st.session_state.form_data['RFM'] = RFM
        else:
            RFM = 25.0  # 默认值
            st.metric("相对脂肪质量 (RFM)", f"{RFM:.2f}", delta="使用默认值")
            st.session_state.form_data['RFM'] = RFM
    
    # 保存数据到session state
    st.session_state.form_data.update({
        'age': age,
        'gender': gender,
        'count': count,
        'height_cm': height_cm,
        'WC_cm': WC_cm,
        'weight_kg': weight_kg
    })
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 返回首页", use_container_width=True):
            st.session_state.current_page = 'welcome'
            st.rerun()
    with col2:
        if st.button("➡️ 下一步：生理指标", type="primary", use_container_width=True):
            st.session_state.current_page = 'physical_indicators'
            st.rerun()

# 生理指标页面
def show_physical_indicators_page():
    st.header("🔬 生理指标")
    st.info("请填写患者的生理测量指标")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("体温 (℃)", min_value=35.2, max_value=38.0, value=36.5, step=0.1, key="temperature")
        pulse = st.number_input("脉搏 (次/分)", min_value=36, max_value=116, value=72, key="pulse")
        systolic = st.number_input("收缩压 (mmHg)", min_value=78, max_value=192, value=120, key="systolic")
        diastolic = st.number_input("舒张压 (mmHg)", min_value=50, max_value=133, value=80, key="diastolic")
        
    with col2:
        left_naked_eye = st.number_input("左眼视力（小数记录法：0.1-2.0）", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="left_naked_eye")
        right_naked_eye = st.number_input("右眼视力（小数记录法：0.1-2.0）", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="right_naked_eye")
        heart_rate = st.number_input("心率 (次/分)", min_value=36, max_value=130, value=72, key="heart_rate")
    
    # 保存数据到session state
    st.session_state.form_data.update({
        'temperature': temperature,
        'pulse': pulse,
        'systolic': systolic,
        'diastolic': diastolic,
        'left_naked_eye': left_naked_eye,
        'right_naked_eye': right_naked_eye,
        'heart_rate': heart_rate
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一步：基本信息", use_container_width=True):
            st.session_state.current_page = 'basic_info'
            st.rerun()
    with col3:
        if st.button("➡️ 下一步：血液生化指标", type="primary", use_container_width=True):
            st.session_state.current_page = 'blood_indicators'
            st.rerun()

# 血液生化指标页面
def show_blood_indicators_page():
    st.header("💉 血液生化指标")
    st.info("请填写患者的血液生化检测指标")
    
    col1, col2 = st.columns(2)
    
    with col1:
        HGB = st.number_input("血红蛋白 (HGB)", min_value=50.0, max_value=234.0, value=135.0, step=1.0, key="HGB")
        WBC = st.number_input("白细胞计数 (WBC)", min_value=1.0, max_value=28.86, value=6.5, step=0.1, key="WBC")
        PLT = st.number_input("血小板计数 (PLT)", min_value=31.0, max_value=906.0, value=250.0, step=1.0, key="PLT")
        glucose = st.number_input("血糖 (glucose)", min_value=1.1, max_value=22.9, value=5.5, step=0.1, key="glucose")
        Creatinine = st.number_input("肌酐 (Creatinine)", min_value=22.0, max_value=761.0, value=70.0, step=1.0, key="Creatinine")
        
    with col2:
        Urea = st.number_input("尿素 (Urea)", min_value=1.0, max_value=41.3, value=5.0, step=0.1, key="Urea")
        TC = st.number_input("总胆固醇 (TC)", min_value=0.88, max_value=13.82, value=4.5, step=0.1, key="TC")
        TG = st.number_input("甘油三酯 (TG)", min_value=0.3, max_value=11.9, value=1.2, step=0.1, key="TG")
        LDL_C = st.number_input("低密度脂蛋白 (LDL_C)", min_value=0.1, max_value=8.19, value=2.5, step=0.1, key="LDL_C")
        HDL_C = st.number_input("高密度脂蛋白 (HDL_C)", min_value=0.1326, max_value=5.0, value=1.2, step=0.1, key="HDL_C")
    
    # 保存数据到session state
    st.session_state.form_data.update({
        'HGB': HGB,
        'WBC': WBC,
        'PLT': PLT,
        'glucose': glucose,
        'Creatinine': Creatinine,
        'Urea': Urea,
        'TC': TC,
        'TG': TG,
        'LDL_C': LDL_C,
        'HDL_C': HDL_C
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一步：生理指标", use_container_width=True):
            st.session_state.current_page = 'physical_indicators'
            st.rerun()
    with col3:
        if st.button("➡️ 下一步：生活习惯", type="primary", use_container_width=True):
            st.session_state.current_page = 'lifestyle'
            st.rerun()

# 生活习惯页面
def show_lifestyle_page():
    st.header("👤 生活习惯与中医体质")
    st.info("请填写患者的生活习惯和中医体质信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**生活习惯**")
        exercise_freq = st.radio("日常是否锻炼", [0, 1], format_func=lambda x: "不锻炼" if x == 0 else "锻炼", horizontal=True, key="exercise_freq")
        smoke = st.radio("日常是否吸烟", [0, 1], format_func=lambda x: "不吸烟" if x == 0 else "吸烟", horizontal=True, key="smoke")
        heart_rhythm = st.radio("心律状况", [0, 1], format_func=lambda x: "齐" if x == 0 else "不齐", horizontal=True, key="heart_rhythm")
        Bscan = st.radio("B超", [0, 1], format_func=lambda x: "正常" if x == 0 else "异常", horizontal=True, key="Bscan")
        
    with col2:
        st.write("**中医体质**")
        tcm_pinghe = st.radio("平和质", [0, 1], format_func=lambda x: "否" if x == 0 else "是", horizontal=True, key="tcm_pinghe")
    
    # 保存数据到session state
    st.session_state.form_data.update({
        'exercise_freq': exercise_freq,
        'smoke': smoke,
        'heart_rhythm': heart_rhythm,
        'Bscan': Bscan,
        'tcm_pinghe': tcm_pinghe
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一步：血液生化指标", use_container_width=True):
            st.session_state.current_page = 'blood_indicators'
            st.rerun()
    with col3:
        if st.button("🚀 开始风险评估", type="primary", use_container_width=True):
            # 执行预测
            perform_prediction()

# 执行预测函数
def perform_prediction():
    if model is not None:
        try:
            # 从session state获取所有数据
            data = st.session_state.form_data
            
            # 按照特征顺序组建输入数组
            # 连续变量部分 (20个)
            continuous_features = np.array([[
                data['age'], data['temperature'], data['pulse'], data['systolic'], data['diastolic'],
                data['left_naked_eye'], data['right_naked_eye'], data['weight_kg'], data['heart_rate'],
                data['HGB'], data['WBC'], data['PLT'], data['glucose'], data['Creatinine'], data['Urea'],
                data['TC'], data['TG'], data['LDL_C'], data['HDL_C'], data['RFM']
            ]])
            
            # 分类变量部分 (7个)
            categorical_features = np.array([[
                data['gender'], data['exercise_freq'], data['smoke'], data['heart_rhythm'], 
                data['Bscan'], data['count'], data['tcm_pinghe']
            ]])
            
            # === Min-Max 归一化处理（仅对连续变量）===
            normalized_continuous = (continuous_features - feature_mins) / (feature_maxs - feature_mins)
            normalized_continuous = np.clip(normalized_continuous, 0, 1)

            # 合并归一化后的连续变量和原始分类变量
            input_features_normalized = np.concatenate([normalized_continuous, categorical_features], axis=1)

            # === 使用最佳阈值进行预测 ===
            prediction_display = None
            high_risk_prob = 0.5

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_features_normalized)
                high_risk_prob = probabilities[0][1]
                
                # 使用自定义阈值而不是模型保存的阈值
                best_threshold = custom_threshold  
                
                if high_risk_prob >= best_threshold:
                    prediction_display = 1
                else:
                    prediction_display = 0
                
                st.info(f"使用决策阈值: {best_threshold:.3f}")
            else:
                # 如果不支持概率预测，直接使用分类预测
                prediction_display = model.predict(input_features_normalized)[0]
                high_risk_prob = 0.8 if prediction_display == 1 else 0.2
            
            # 显示预测结果
            st.subheader("📊 风险评估结果")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # 根据调整后的阈值显示结果
                if prediction_display == 0:
                    st.success("✅ 风险评估: 低风险")
                    st.metric("风险等级", "低风险", delta="良好")
                else:
                    st.error("⚠️ 风险评估: 高风险")
                    st.metric("风险等级", "高风险", delta="需关注", delta_color="inverse")
            
            with result_col2:
                # 显示预测概率
                low_risk_prob = (1 - high_risk_prob) * 100
                high_risk_prob_display = high_risk_prob * 100
                
                st.metric("低风险概率", f"{low_risk_prob:.1f}%")
                st.metric("高风险概率", f"{high_risk_prob_display:.1f}%")
                
                # 显示概率条
                st.progress(int(high_risk_prob_display))
                st.caption(f"风险概率: {high_risk_prob_display:.1f}%")
                st.caption(f"决策阈值: {best_threshold*100:.1f}%")
            
            # 建议信息
            st.info("💡 **建议**: 请结合临床实际情况综合判断，必要时进行进一步检查。")
            
            # 添加重新评估按钮
            st.markdown("---")
            if st.button("🔄 重新开始评估", type="secondary", use_container_width=True):
                st.session_state.current_page = 'welcome'
                st.session_state.form_data = {}
                st.rerun()
            
        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")
    else:
        st.error("模型未正确加载，无法进行预测。")

# 批量预测功能
def show_batch_prediction():
    st.markdown("---")
    st.header("📁 批量预测")

    uploaded_file = st.file_uploader("上传CSV文件进行批量预测", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.subheader("数据预览")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # 检查列名
            expected_columns = all_feature_names
            missing_columns = [col for col in expected_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"缺少必要的列: {missing_columns}")
            else:
                if st.button("执行批量风险评估", use_container_width=True):
                    with st.spinner('正在处理批量预测...'):
                        # 分离连续变量和分类变量
                        batch_continuous = batch_data[continuous_feature_names].values
                        batch_categorical = batch_data[categorical_feature_names].values
                        
                        # 对连续变量进行归一化
                        batch_normalized = (batch_continuous - feature_mins) / (feature_maxs - feature_mins)
                        batch_normalized = np.clip(batch_normalized, 0, 1)
                        
                        # 合并特征
                        batch_features = np.concatenate([batch_normalized, batch_categorical], axis=1)
                        
                        # 使用自定义阈值进行预测
                        if hasattr(model, 'predict_proba'):
                            batch_probabilities = model.predict_proba(batch_features)
                            batch_high_risk_probs = batch_probabilities[:, 1]
                            # 使用内置阈值进行预测
                            batch_predictions = (batch_high_risk_probs >= custom_threshold).astype(int)
                            
                            batch_data['预测风险等级'] = batch_predictions
                            batch_data['预测风险等级'] = batch_data['预测风险等级'].map({0: '低风险', 1: '高风险'})
                            batch_data['低风险概率'] = [f"{prob:.4f}" for prob in batch_probabilities[:, 0]]
                            batch_data['高风险概率'] = [f"{prob:.4f}" for prob in batch_probabilities[:, 1]]
                        else:
                            batch_predictions = model.predict(batch_features)
                            batch_data['预测风险等级'] = batch_predictions
                            batch_data['预测风险等级'] = batch_data['预测风险等级'].map({0: '低风险', 1: '高风险'})
                        
                        st.subheader("批量预测结果")
                        st.dataframe(batch_data, use_container_width=True)
                        
                        # 统计结果
                        risk_counts = batch_data['预测风险等级'].value_counts()
                        st.write("**风险分布统计:**")
                        st.write(risk_counts)
                        
                        # 下载结果
                        import io
                        output = io.BytesIO()
                        batch_data.to_csv(output, index=False, encoding='utf-8-sig')
                        csv_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 下载预测结果",
                            data=csv_data,
                            file_name="心血管代谢疾病风险评估结果.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
        except Exception as e:
            st.error(f"处理上传文件时出错: {e}")

# 主页面路由
def main():
    # 根据当前页面状态显示相应内容
    if st.session_state.current_page == 'welcome':
        show_welcome_page()
    elif st.session_state.current_page == 'basic_info':
        show_basic_info_page()
    elif st.session_state.current_page == 'physical_indicators':
        show_physical_indicators_page()
    elif st.session_state.current_page == 'blood_indicators':
        show_blood_indicators_page()
    elif st.session_state.current_page == 'lifestyle':
        show_lifestyle_page()
    
    # 只有在欢迎页面显示批量预测
    if st.session_state.current_page == 'welcome':
        show_batch_prediction()

    # 侧边栏信息
    st.sidebar.header("📖 使用说明")
    st.sidebar.info("""

    **评估流程:**
    - 👤 基本信息: 年龄、性别、疾病数量、身体测量指标
    - 🔬 生理指标: 体温、脉搏、血压、视力、心率
    - 💉 血液生化指标: 血常规、血脂、血糖等指标  
    - 👤 生活习惯: 生活方式、心律、B超、中医体质

    **视力对照表:**
    - 对数视力表（5分制）与小数记录法（国际标准）对照:
    - 5.3 = 2.0
    - 5.2 = 1.5
    - 5.1 = 1.2
    - 5.0 = 1.0
    - 4.9 = 0.8
    - 4.8 = 0.6
    - 4.7 = 0.5
    - 4.6 = 0.4
    - 4.5 = 0.3
    - 4.4 = 0.25
    - 4.3 = 0.2
    - 4.2 = 0.15
    - 4.1 = 0.12
    - 4.0 = 0.1
    """)

    st.sidebar.header("🔍 特征信息")
    st.sidebar.metric("总特征数", "27", "20连续 + 7分类")

if __name__ == "__main__":
    main()
import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

with open('model/cleaned_data.pkl','rb') as file:
    data = pickle.load(file)

model = pickle.load(open('model/model.pkl','rb'))


def add_sidebar():
    st.sidebar.header('Personalize Your Well-Being Journey')

    cat_features = [col for col in data.columns if data[col].dtype=='object']
    catcols_high_cardinality = [col for col in cat_features if data[col].nunique()>10]
    catcols_low_cardinality = [col for col in cat_features if data[col].nunique()<=10]
    num_features = ['Age']
    cat_num_features = ['Work Pressure','Job Satisfaction','Work/Study Hours','Financial Stress']
    input_dict = {}
    
    for i in catcols_low_cardinality:
        input_dict[i] = st.sidebar.radio(
            label = i,
            options=[items for items in data[i].unique()]
        )
    
    for i in catcols_high_cardinality:
        input_dict[i] = st.sidebar.selectbox(
            label = i,
            options = [items for items in data[i].unique()],
            index = None
        )

    for i in num_features:
        input_dict[i] = st.sidebar.slider(
            label = i,
            min_value = 10,
            max_value = 100,
            step = 1,
            value = data[i].min() ,
        )
    
    for i in cat_num_features:
        input_dict[i] = st.sidebar.select_slider(
            label = i,
            options = sorted([items for items in data[i].unique()]),
            value = data[i].min()
        )

    return input_dict

def add_predictions(input_data,model):
    te = pickle.load(open('model/target_encoder.pkl','rb'))
    oe = pickle.load(open('model/ordinal_encoder.pkl','rb'))

    df = pd.DataFrame([input_data])
    new_cols = ['Gender','Age','Working Professional or Student','Work Pressure','Job Satisfaction','Sleep Duration','Dietary Habits','Degree','Have you ever had suicidal thoughts ?','Work/Study Hours','Financial Stress','Family History of Mental Illness']
    df_reindex = df.reindex(columns=new_cols)
    
    catcols_high_cardinality = ['Sleep Duration', 'Dietary Habits', 'Degree']
    catcols_low_cardinality = ['Gender', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    
    df_reindex[catcols_low_cardinality] = oe.transform(df_reindex[catcols_low_cardinality])
    df_reindex[catcols_high_cardinality] = te.transform(df_reindex[catcols_high_cardinality])
    
    input_array = df_reindex.to_numpy()
    
    pred = model.predict(input_array)
    if pred[0] == 0:
        st.write(':green[**You are not Depressed**]')
    else:
        st.markdown(':red[**You are Depressed**]')
        st.write('Consult a Psychologists Asap')

    
    prob = (model.predict_proba(input_array)[0][1])
    st.write('Probability of Depression: ', prob*100,'%')

    feature_imp = model.feature_importances_

    return feature_imp,prob
    # st.write(pred)

    categories = ['Work Pressure','Job Satisfaction','Work/Study Hours','Financial Stress']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            inp['Work Pressure'],inp['Job Satisfaction'],inp['Work/Study Hours'],inp['Financial Stress']
        ],
        theta=categories,
        fill='toself',
        name=''
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 12]
        )),
        showlegend=False
        )

    return fig

def get_real_time_prob(depression_probability):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Depression Probability"],
        y=[depression_probability],
        marker_color='crimson'
    ))

    fig.update_layout(
        title="Real-Time Depression Probability",
        yaxis=dict(range=[0, 1])
    )
    return fig

def data_analysis(train_df):

    contingency_table1 = pd.crosstab(train_df['Gender'], train_df['Depression'])
    contingency_table2 = pd.crosstab(train_df['Family History of Mental Illness'], train_df['Depression'])
    contingency_table3 = pd.crosstab(train_df['Have you ever had suicidal thoughts ?'], train_df['Depression'])
    contingency_table4 = pd.crosstab(train_df['Working Professional or Student'], train_df['Depression'])


    fig1,ax = plt.subplots(2,2,figsize=(22,10))
    sns.heatmap(contingency_table1,annot=True,fmt='d',cmap='viridis',ax=ax[0,0])
    sns.heatmap(contingency_table2,annot=True,fmt='d',cmap='viridis',ax=ax[0,1])
    sns.heatmap(contingency_table3,annot=True,fmt='d',cmap='viridis',ax=ax[1,0])
    sns.heatmap(contingency_table4,annot=True,fmt='d',cmap='viridis',ax=ax[1,1])

    contingency_table1 = pd.crosstab(train_df['Work Pressure'], train_df['Depression'])
    contingency_table2 = pd.crosstab(train_df['Job Satisfaction'], train_df['Depression'])

    fig2,ax = plt.subplots(1,3,figsize=(22,10))
    sns.heatmap(contingency_table1,annot=True,fmt='d',cmap='viridis',ax=ax[0])
    sns.heatmap(contingency_table2,annot=True,fmt='d',cmap='viridis',ax=ax[1])
    sns.boxplot(y='Age', x='Depression', data=train_df,ax=ax[2])

    fig, ax = plt.subplots(1,2,figsize=(12.5,5))
    sns.lineplot(x='Financial Stress',y='Age',data=train_df,ax=ax[0])
    contingency_table4 = pd.crosstab(train_df['Financial Stress'], train_df['Depression'])
    sns.heatmap(contingency_table4,annot=True,fmt='d',cmap='viridis',ax=ax[1])
    
    return fig1,fig2,fig

def insights(input_data):
    # st.write(input_data)
    if input_data['Sleep Duration'] != '7-8 hours' and input_data['Sleep Duration'] !='More than 8 hours':
        st.markdown(' - Increase your sleep hours to 7-8 hours to decrease the risk of Depression. \n - Avoid Late Night Works.')
    if input_data['Dietary Habits'] != 'Healthy':
        st.markdown('- Start eating healthy food. *Remember Health is Wealth*.')
    if input_data['Have you ever had suicidal thoughts ?'] == 'Yes':
        st.markdown('- Your pain is valid, but it does not have to be permanent—there is help, hope, and people who care deeply about you \n- **Suicide Prevention Helpline Number : 91-9820466726** ')
    if input_data['Work Pressure'] >= 3:
        st.markdown("- Take one task at a time, set clear boundaries, and remember it's okay to say no when you're overwhelmed")
    if input_data['Job Satisfaction'] > 3:
        st.markdown("- It's okay to feel unsatisfied with your job—use it as motivation to explore new opportunities and work towards something better")
    if input_data['Financial Stress'] > 3:
        st.markdown("- Start creating a budget, prioritizing essentials, and seeking professional financial advice if needed.")
    if input_data['Work/Study Hours'] > 5:
        st.markdown("- Set realistic goals, focus on time management, and allocate breaks to maintain a healthy balance between study/work and personal life.")
        
def plots(data,feature_imp,prob):

    real_time_prob = get_real_time_prob(prob)
    st.plotly_chart(real_time_prob)


    feature_imp_df = pd.DataFrame({
        'Feature' : data.drop(columns = ['Depression'],axis=1).columns,
        'Importance' : feature_imp
    })
    feature_imp_df = feature_imp_df.sort_values(by='Importance',ascending=False)
    # st.write(feature_imp_df)
    fig = px.bar(feature_imp_df, x='Feature', y='Importance', title="Feature Importance in Mental Health Prediction",
             labels={'Importance': 'Feature Importance'}, color='Importance')

    fig.update_layout(template="plotly_dark", xaxis_title="Features", yaxis_title="Importance")
    st.plotly_chart(fig)

    
    st.subheader('Here are some plots presenting the data analysis of various features in relation to depression')
    st.text('Note - These plots are made using statistical data obtained from Kaggle Dataset on Mental Health. \nLink to original dataset is provided below.')
    fig1,fig2,fig = data_analysis(data)
    st.divider()
    st.text('These plots illustrates the relation between Features like Gender, Family History of depression, etc. and the no. of people suffering from Depression.')
    st.pyplot(fig1)
    st.text('Inference Drawn : \n1. Trends reveals that Males are generally more to Depression than Females \n2. A person having suicidal thoughts is suffering from depression \n3. There have been reported more Depression cases in Students compared to working professionals \n4. A person having a family history of mental illness is likely to suffer from depression' )
    st.divider()
    st.text('These heatmaps illustrates the relationship between work pressure and job satisfaction vs Cases of Depression  ')
    st.pyplot(fig2)
    st.text('Inference Drawn : \n1. Depression cases are directly proportional to work pressure \n2. A person unsatisfied by his/her Job has a high probability of Depression.')
    st.divider()
    st.text('These plots describe how financial stress is related to age and depression')
    st.pyplot(fig)
    st.text('Inference Drawn: \n1. It is seen that people of less age have hign financial stress \n2. More financial stress leads to increase in chances of depression')
    st.divider()

def main():
    st.set_page_config(
        page_title = 'Mental Health Predictor',
        page_icon = 'brain:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()
    
    
    with st.container():
        st.title('Mindful Insights: Understanding Your Mental Health')
        st.write('Depression affects millions globally and can stem from various factors like genetics, environment, and stress. Understanding it is the first step toward improving mental well-being. By recognizing the signs early, individuals can take proactive steps to seek help and build resilience. Remember, mental health is a journey, and reaching out is a sign of strength, not weakness.')

    col1,col2 = st.columns([4,1.5])
    
    with col2:
        st.subheader('Checkup: Are You at Risk of Depression?')
        st.write('Input your personal Information in the sidebar for predictions')

        feature_imp,prob = add_predictions(input_data,model)
        st.divider()
        st.subheader('Here are some recommendations to improve your mental health :')
        insights(input_data)
        
    
    with col1:
        st.subheader('Empower Your Well-Being with Data-Driven Insights')
        plots(data,feature_imp,prob)
        st.markdown('Dataset used to train the model : https://drive.google.com/file/d/14rVIkd-hQD6ggBA-v7vr6XmPnCzHQfr-/view?usp=sharing')

main()

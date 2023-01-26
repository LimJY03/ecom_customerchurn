import streamlit as st
import time
import pandas as pd
from PIL import Image
import sklearn
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

clf = joblib.load('finalized_model.sav')

with st.sidebar:
    st.subheader('E-commerce Customer Churn Application :shopping_trolley:')

side_menu = st.sidebar.selectbox('Select Page',['Home','User Manual', 'Analysis and Prediction','About'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('*Copywrite by 2023*')

if side_menu =='Home':
    st.title('E-commerce Customer Churn Analysis and Prediction Application')
    st.markdown('### This app shows how to analyse and predict customer churn')
    customer = Image.open('customer.jpg')
    st.image(customer, use_column_width=None)

if side_menu =='User Manual':
    st.title('User Manual :pushpin:')
    st.markdown('---')

    st.subheader('Choice 1: Predict Customer Churn Using Manual Insert')
    st.write('1. Go to "Analysis and Prediction Page"')
    st.write('2. Choose "Manual" for the "Select An Option" box')
    st.write('3. Key in or choose all the details, then you can view your final input data as a table below "Customer Details"')
    st.write('4. Press the button "Predict Customer Churn & Probability", it will takes a while to load the result')
    st.write('5. You get your result! You will also be able to see a graph of that customer churning probability')

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Choice 2: Predict Customer Churn Using File')
    st.write('1. Go to "Analysis and Prediction Page"')
    st.write('2. Choose "File" for the "Select An Option" box')
    st.write('3. Upload your csv file, it will takes some time to clean and process your file')
    st.write('4. Choose a chart you wish to explore')
    st.write('5. After visualizing data, you can select Customer ID that you wish to predict churning probability')
    st.write('6. You get your result! You will also be able to see a graph of that customer churning probability')


if side_menu=='Analysis and Prediction':
    st.title('Churn Analysis & Prediction :bar_chart:')
    st.write('### How would you like to predict?')
    add_selectbox = st.selectbox(':point_down:',['Select An Option','Manual', 'File'])
    if add_selectbox == 'Select An Option':
        pass
    if add_selectbox == 'Manual':
        st.write('---')

        st.write('### Key In:')
        def user_input_feature():
            gender = st.selectbox('**Gender:**', ['Male', 'Female'])
            martialstatus = st.selectbox('**Martial Status:**', ['Single','Married','Divorced'])
            tenure = st.number_input('**Number of months the customer has been with the current E-commerce app:**',
                                     min_value=0, max_value=240, value=0)
            citytier = st.number_input('**Customer Current City Tier (Tier 1 = most developed, Tier 2 = moderately developed, Tier 3 = most underdeveloped):**',
                                     min_value=1, max_value=3, value=1)
            numberofaddress = st.slider('**Customer have how many address:**',min_value=0, max_value=10, value=0)
            preferredlogindevice = st.selectbox('**Most Used Device Type:**', ['Phone','Computer'])
            numberofdeviceregistered = st.slider('**Customer have how many device:**',min_value=0, max_value=10, value=0)
            hourspendonapp = st.number_input('**Average hours spend a day with the current E-commerce app:**',
                                     min_value=0, max_value=24, value=0)
            preferredordercat = st.selectbox('**Category that the customer bought most:**', ['Phone','Laptop & Accessory','Fashion','Grocery','Others'])
            preferredpaymentmode = st.selectbox('**Preferred payment method of customer:**',['Debit Card','Credit Card','E wallet','Cash on Delivery','Unified Payments Interface(UPI)'])
            warehousetohome = st.number_input('**Distance in between warehouse to home of customer(in meter):**',
                                              min_value=0, max_value=500, value=0)
            daysincelastorder = st.number_input('**How many days since last order by customer:**',
                                     min_value=0, max_value=90, value=0)
            ordercount = st.slider('**Total number of orders has been places by customer in last month:**',
                                   min_value=0, max_value=10, value=0)
            couponused = st.slider('**Total number of coupon used by customer in last month:**',
                                   min_value=0, max_value=20, value=0)
            cashbackamount = st.number_input('**Average cashback by customer in last month:**',
                                     min_value=0, max_value=500, value=0)
            complain = st.number_input('**How many complaints has been raised by customer in last month:**',
                                     min_value=0, max_value=10, value=0)
            orderamounthikefromlastyear = st.number_input('**Percentage increase in orders by customer from last year (%):**',
                                     min_value=0, max_value=100, value=0)
            satisfactionscore = st.slider('**Satisfactory score of customer on current E-commerce service (0-lowest, 5-highest):**',
                                   min_value=0, max_value=5, value=0)
            input_dict = {
                "gender": gender,
                "martialstatus": martialstatus,
                "tenure": tenure,
                "citytier": citytier,
                "numberofaddress": numberofaddress,
                "preferredlogindevice": preferredlogindevice,
                "numberofdeviceregistered": numberofdeviceregistered,
                "hourspendonapp": hourspendonapp,
                "preferredordercat": preferredordercat,
                "preferredpaymentmode": preferredpaymentmode,
                "warehousetohome": warehousetohome,
                "daysincelastorder": daysincelastorder,
                "ordercount": ordercount,
                "couponused": couponused,
                "cashbackamount": cashbackamount,
                "complain": complain,
                "orderamounthikefromlastyear": orderamounthikefromlastyear,
                "satisfactionscore": satisfactionscore,
            }

            features = pd.DataFrame(input_dict, index=[0])
            return features

        df1 = user_input_feature()

        st.write('---')
        st.subheader('Customer Details:')
        st.write(df1)

        enc = LabelEncoder()
        for col in df1.select_dtypes(include='object'):
            df1[col] = enc.fit_transform(df1[col])

        output = 0
        output_prob = 0
        if st.button('Predict Customer Churn & Probability'):
            X = df1
            y_pred = clf.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Churn: {0}, Churn Probability: {1}'.format(output, output_prob))
            st.write('')
            st.subheader('Churn Probability Graph:')
            labels = 'churn_prob', 'notChurn_prob'
            sizes = [output_prob*100, 100-output_prob*100]

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)

    if add_selectbox == 'File':
        file_upload = st.file_uploader('Upload csv file for predictions', type=['csv'])
        if file_upload is not None:
            df2 = pd.read_csv(file_upload)
            st.write(df2)
            st.write('Total number of rows of missing values:', df2.isnull().any(axis=1).sum())
            with st.spinner('Wait for cleaning...'):
                time.sleep(3)
            for i in df2.columns:
                if df2[i].isnull().sum() > 0:
                    df2[i].fillna(df2[i].mean(), inplace=True)
            st.write('Total number of missing values:',df2.isnull().any(axis=1).sum())
            st.write('Data cleaned')
            st.write('')


            st.subheader('Visualize Your Data')
            add_select = st.selectbox('Choose a value:',['Tenure', 'Order Count','Days Since Last Order','Increase In Orders','Hours Spend on App'])

            if add_select == 'Tenure':
                fig2, ax2 = plt.subplots()
                ax2.hist(x='Tenure', data= df2, color='b',edgecolor='black', linewidth=1.2)
                plt.title("Distribution of Tenure of the Customers on the platform")
                plt.xlabel('Tenure')
                plt.ylabel('Count')
                st.pyplot(fig2)

            if add_select == 'Order Count':
                fig3, ax3 = plt.subplots()
                ax3.hist(x='OrderCount', data=df2, color='c',edgecolor='black', linewidth=1.2)
                plt.title("Distribution on Total Number of Orders in Last Month")
                plt.xlabel('Orders')
                plt.ylabel('Count')
                st.pyplot(fig3)

            if add_select == 'Days Since Last Order':
                fig4, ax4 = plt.subplots()
                ax4.hist(x='DaySinceLastOrder', data=df2, color='m',edgecolor='black', linewidth=1.2)
                plt.title("Distribution of Recency of the Customers")
                plt.xlabel('Days')
                plt.ylabel('Count')
                st.pyplot(fig4)

            if add_select == 'Increase In Orders':
                fig5, ax5 = plt.subplots()
                ax5.hist(x='OrderAmountHikeFromlastYear', data=df2, color='r',edgecolor='black', linewidth=1.2)
                plt.title("Distribution of Percentage of Customer Increase In Orders")
                plt.xlabel('Orders Hike')
                plt.ylabel('Count')
                st.pyplot(fig5)

            if add_select == 'Hours Spend on App':
                fig6, ax6 = plt.subplots()
                ax6.hist(x='HourSpendOnApp', data=df2, color='y',edgecolor='black', linewidth=1.2)
                plt.title("Distribution of Hours Spent on E-commerce Application by Customers")
                plt.xlabel('Hours Spend')
                plt.ylabel('Count')
                st.pyplot(fig6)

            st.subheader('Predict Customer Churn by Customer ID')
            selected_id = st.selectbox('Select:', tuple(df2['CustomerID']))

            enc = LabelEncoder()
            for col in df2.select_dtypes(include='object'):
                df2[col] = enc.fit_transform(df2[col])

            X = df2[df2['CustomerID'] == selected_id].iloc[0, 2:21].values
            y_pred = clf.predict_proba(X.reshape(1, -1))[0, 1]
            churn = y_pred >= 0.5
            churn_prob = float(y_pred)
            churn = bool(churn)
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Churn: {0}, Churn Probability: {1}'.format(churn, churn_prob))
            st.write('')
            st.subheader('Churn Probability Graph:')
            labels = 'churn_prob', 'notChurn_prob'
            sizes = [churn_prob * 100, 100 - churn_prob * 100]

            fig2, ax2 = plt.subplots()
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig2)

if side_menu=='About':
    st.title('Credits :sparkles:')
    st.subheader('Author :female-student:')
    st.write('Copywrite of Yeo Jie Hui')
    st.write('Third Year Data Science, University of Malaya Student')
    linkedin= 'https://www.linkedin.com/in/yeo-jie-hui-786ba3215/'
    st.write('Feel Free to Connect Me in LinkedIN:', linkedin)
    st.write('')
    st.subheader('About dataset :bookmark_tabs:')
    st.write('Data source: [Kaggle] (https://www.kaggle.com/code/ankitverma2010/e-commercecustomerchurn/data)')

    st.subheader('Link to Code :open_file_folder:')
    st.write('This is the link to my source code: [Github]')
    st.write('')
    st.write('')

    st.markdown('If you have any questions or feedback on my application, reach out to me through LinkedIN!')
    st.write('Enjoy and Play Around This Application, Hope You Like It! :laughing:')






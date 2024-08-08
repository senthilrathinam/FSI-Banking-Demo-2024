import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.config import Config
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
import boto3
import bank_app_lib as glib #reference to local lib script


# Set the page configuration
st.set_page_config(page_title="Bank Peer Analytics (Risk Metrics Scoring)", layout="wide")

# Custom CSS for fonts, text styling, centering the title in a light blue ribbon, and left sidebar menu
st.markdown(
    """
    <style>
    .main, .sidebar .sidebar-content, .sidebar .block-container {
        background-color: #ADD8E6;
        color: black;
        font-family: 'Times New Roman', serif;
        font-size: 8px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 8px;
    }
    .stRadio>label, .stSelectbox>label, .stTextInput>label, .stTextArea>label, .stSlider>label, .stCheckbox>label {
        color: black;
        font-size: 16px;
    }
    .sidebar .sidebar-content h1 {
        font-size: 18px;
        color: blue;
    }
    .sidebar .sidebar-content img {
        width: 25%;
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    .title-container {
        background-color: #333333;
        border-top: 2px solid #4DA6FF;
        border-bottom: 2px solid #4DA6FF;
        padding: 10px 0;
        text-align: center;
    }
    .title-container h1 {
        font-size: 36px;
        color: white;
        margin: 0;
    }
    .title-container p:nth-child(2) {
        font-size: 16px;
        color: white;
    }
    .title-container p:nth-child(3) {
        font-size: 16px;
        color: white;
    }
    .highlight {
        color: #4DA6FF;
    }
    .frame-container {
        background-color: red;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .custom-file-uploader .stFileUploader {
        max-width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
    .stSidebar>div:first-child {
        font-family: 'Times New Roman', serif;
        font-size: 8px;
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Light blue ribbon with centered page title
st.markdown(
    """
    <div class="title-container">
        <h1>👋 Hey There! I am <span class="highlight">BankIQ</span>⚡</h1>
        <p>I answer questions after reading documents, webpages, images with text, YouTube videos, audio files and spreadsheets.</p>
        <p>✨ You can ask me anything ✨</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize S3 client
s3 = boto3.client('s3', config=Config(signature_version='s3v4'))

# Initialize S3 bucket names
base_bnk_s3_bucket = "banking-demo-base-bank-sr"
peer_bnk_s3_bucket = "banking-demo-peer-bank-sr"
metrics_summary_s3_bucket = "banking-demo-metrics-summary-sr"


# Initialize bedrock components
session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client
bedrock_model_id = "amazon.titan-text-express-v1" #set the foundation model


def display_sample_msg(prompt):
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0,  
            "topP": 0.5,
            "maxTokenCount": 1024,
            "stopSequences": []
        }
    }) #build the request payload
    
    #
    response = bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept='application/json', contentType='application/json') #send the payload to Amazon Bedrock
    
    #
    response_body = json.loads(response.get('body').read()) # read the response
    
    response_text = response_body["results"][0]["outputText"] #extract the text from the JSON response
    
    return response_text
    #print(response_text)





def upload_to_s3(file, bucket_name, object_name):
    try:
        s3.upload_fileobj(file, bucket_name, object_name)
        return True
    except NoCredentialsError:
        st.error("Credentials not available")
        return False


def list_folders_in_s3(bucket_name, prefix=''):
    try:
        print('bucket_name', bucket_name)
        print('prefix', prefix)
        response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/', Prefix=prefix)
        # print('response', response)
        if 'CommonPrefixes' in response:
            print(prefix['Prefix'].rstrip('/') for prefix in response['CommonPrefixes'])
            return [prefix['Prefix'].rstrip('/') for prefix in response['CommonPrefixes']]
        else:
            return []
    except NoCredentialsError:
        st.error("Credentials not available")
        return []

def list_folders_in_s3_old_with_root_path(bucket_name, prefix=''):
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/', Prefix=prefix)
        folders = []
        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                folder_name = folder['Prefix'].rstrip('/')
                folders.append(folder_name)
        return folders
    except NoCredentialsError:
        st.error("Credentials not available")
        return []


def list_folders_in_s3_new(bucket_name, prefix=''):
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/', Prefix=prefix)
        folders = []
        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                folder_name = folder['Prefix'].rstrip('/')
                if '/' in folder_name:
                    # If the folder name contains a '/', it's a second-level subfolder
                    folders.append(folder_name.split('/')[-1])
                else:
                    # Recursively call the function to get second-level subfolders
                    sub_folders = list_folders_in_s3_new(bucket_name, folder_name + '/')
                    folders.extend(sub_folders)
        return folders
    except NoCredentialsError:
        st.error("Credentials not available")
        return []

def list_files_in_s3(bucket_name, prefix):
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
    except NoCredentialsError:
        st.error("Credentials not available")
        return []

def download_file_from_s3(bucket_name, file_key):
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        return obj['Body'].read()
    except NoCredentialsError:
        st.error("Credentials not available")
        return None

def get_latest_file_from_s3(bucket_name, prefix):
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        print("bucket name", bucket_name)
        print("bucket prefix", prefix)
        
        print("inside get files from s3 method: ", response)
        if 'Contents' in response:
            print("inside contents if")
            files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            latest_file = files[0]['Key']
            return latest_file
        else:
            print("inside contents else")
            return None
    except NoCredentialsError:
        st.error("Credentials not available")
        return None

# Function for Financial Reports Hub
def financial_reports_hub():
    #change the below code to display the subheader at the cetner of the page
    st.markdown('<hr style="border:2px solid black"> </hr>', unsafe_allow_html=True)

    # Center-align the subheader
    html_subheader = """
    <div style="text-align: center;">
        <h3>Financial Reports Hub</h3>
    </div>
    """
    st.markdown(html_subheader, unsafe_allow_html=True)

    # st.text('test')
    st.markdown('<hr style="border:2px solid black"> </hr>', unsafe_allow_html=True)

    # st.markdown('<div class="frame-container">', unsafe_allow_html=True)

    # Manual File Uploader
    st.subheader("File Uploader")
    col1, col2 = st.columns([1, 1])
    with col1:
        bank_selection = st.radio("Select a Bank", ['Base Bank', 'Peer Bank'])
    with col2:
        report_type = st.radio("Select Report Type", ['10-K Reports', '10-Q Reports'])

    # File Uploader
    st.markdown('<div class="custom-file-uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(f"Upload {report_type}", key=f"{bank_selection.lower()}_{report_type.lower()}")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if bank_selection == 'Base Bank':
            # bucket_name = "banking-demo-day-base-bank"
            bucket_name = base_bnk_s3_bucket
        else:
            #bucket_name = "banking-day-demo-teamshashi"
            bucket_name = peer_bnk_s3_bucket
        object_name = f"{bank_selection}/{report_type}/{uploaded_file.name}"

        if st.button("Upload"):
            if upload_to_s3(uploaded_file, bucket_name, object_name):
                st.success(f"{uploaded_file.name} uploaded successfully to {bucket_name}/{object_name}!")
                st.stop()  # Stop execution after successful upload
            else:
                st.error("File upload failed")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:2px solid black"> </hr>', unsafe_allow_html=True)

    # Display files from S3
    st.subheader("Files in Repository")
    file_type = st.radio("Select File Type", ["Base Bank Files", "Peer Bank Files"])

    if file_type == "Base Bank Files":
        if st.button("Display Base Bank Files"):
            # base_bucket_name = base_bnk_s3_bucket # Replace with your Base Bank S3 bucket name
            # files = list_files_in_s3(base_bucket_name, "")
            # if files:
            #     st.markdown("### Base Bank Files")
            #     for file in files:
            #         st.markdown(f"- {file}")
            # else:
            #     st.markdown("No files found for Base Bank Files")
            base_bucket_name = base_bnk_s3_bucket
            files = list_files_in_s3(base_bucket_name, "")
            if files:
                st.markdown("### Base Bank Files")
                cols = st.columns(3)

                with cols[0]:
                    st.subheader("First-level Folders")
                    first_level_folders = set([file_path.split('/')[0] for file_path in files])
                    for folder in first_level_folders:
                        st.write(folder)

                with cols[1]:
                    st.subheader("Second-level Folders")
                    second_level_folders = set(['/'.join(file_path.split('/')[1:2]) for file_path in files if len(file_path.split('/')) > 2])
                    for folder in second_level_folders:
                        st.write(folder)

                with cols[2]:
                    st.subheader("Files")
                    for file_path in files:
                        folder_levels = file_path.split('/')
                        if len(folder_levels) > 2:
                            st.write(folder_levels[-1])

            else:
                st.markdown("No files found for Base Bank Files")


    else:
        peer_bank_folders = list_folders_in_s3(peer_bnk_s3_bucket, 'Peer Bank/' )
        # st.write("i am here")

        peer_bank_selection = st.selectbox("Select a Peer Bank", peer_bank_folders)
        if st.button("Display Peer Bank Files"):
            peer_bucket_name = peer_bnk_s3_bucket 
            prefix = f"{peer_bank_selection}/"
     
            files = list_files_in_s3(peer_bucket_name, prefix)
            if files:
                st.markdown(f"### {peer_bank_selection} Files")
                cols = st.columns(3)

                with cols[0]:
                    st.subheader("First-level Folders")
                    first_level_folders = set([file_path.split('/')[0] for file_path in files])
                    for folder in first_level_folders:
                        st.write(folder)

                with cols[1]:
                    st.subheader("Second-level Folders")
                    second_level_folders = set(['/'.join(file_path.split('/')[1:2]) for file_path in files if len(file_path.split('/')) > 2])
                    for folder in second_level_folders:
                        st.write(folder)

                with cols[2]:
                    st.subheader("Files")
                    for file_path in files:
                        folder_levels = file_path.split('/')
                        if len(folder_levels) > 2:
                            st.write(folder_levels[-1])

            else:
                st.markdown(f"No files found for {peer_bank_selection} Files")



    st.markdown('<hr style="border:2px solid black"> </hr>', unsafe_allow_html=True)

    html_message = """
    <div style="text-align: center;">
        <p><em>Peer Bank Analytics - Powered by Amazon Bedrock</em></p>
    </div>
    """
    st.markdown(html_message, unsafe_allow_html=True)



def plot_metric_trend(df, metric_column, metric_name):
    base_bank = df[df['Bank Type'] == 'Base']
    peer_banks = df[df['Bank Type'] == 'Peer']

    # Create a vertical stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    banks = df["Bank"].unique()
    quarters = df["Quarter"].unique()
    
    # Stack the bars by bank
    stacked_data = df.groupby(["Bank", "Quarter"])[metric_column].sum().unstack("Quarter")
    
    # Create the bar plot
    bars = stacked_data.plot(kind="bar", ax=ax, stacked=True)
    
    # Add value labels to the bars
    for bar in ax.containers:
        ax.bar_label(bar, fmt="%.1f", label_type="center")
    
    ax.set_title(f"{metric_name} by Bank Type and Quarter")
    ax.set_xlabel("Bank")
    ax.set_ylabel(metric_name)
    # ax.legend(labels=quarters, title="Quarter", loc='center left')
    ax.legend(labels=quarters, title="Quarter", loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_vertical_bar_chart_common(df, period, metric_column, metric_name):
    # Create a vertical bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    banks = df["Bank"].unique()
    periods = df[period].unique()

    # Group the data by quarter and bank
    grouped_data = df.groupby([period, "Bank"])[metric_column].sum().reset_index()

    # Plot the bars
    bar_width = 0.25
    bar_positions = [i for i in range(len(periods))]
    bars = []

    for i, bank in enumerate(banks):
        bank_data = grouped_data[grouped_data["Bank"] == bank]
        bar = ax.bar([x + bar_width * i for x in bar_positions], bank_data[metric_column], width=bar_width, label=bank)
        bars.extend(bar)
        for rect in bar:
            height = rect.get_height()
            ax.annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_title(f"{metric_name} by {period} and Bank")
    ax.set_xlabel(period)
    ax.set_ylabel(metric_name)
    ax.set_xticks([x + bar_width for x in bar_positions], periods)
    ax.legend(labels=banks, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the layout to make space for the legend
    plt.subplots_adjust(right=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    

# Function for Analyze Metrics - Summary
def analyze_metrics_summary():
    # List metric folders from S3 bucket
    metric_folders = list_folders_in_s3_new(metrics_summary_s3_bucket, 'Quarterly/')
    metric = st.selectbox("Select a Metric", metric_folders)

    # Generate Graph and Summary
    if st.button("Create Graph & Summary"):
        if metric:
            yearly_data = []
            for year_folder in list_folders_in_s3_new(metrics_summary_s3_bucket, f'Quarterly/{metric}/'):
                quarterly_data = []
                for quarter_folder in list_folders_in_s3_new(metrics_summary_s3_bucket, f'Quarterly/{metric}/{year_folder}/'):
                    file_key = f"Quarterly/{metric}/{year_folder}/{quarter_folder}/{metric}.xlsx"
                    file_content = download_file_from_s3(metrics_summary_s3_bucket, file_key)
                    if file_content:
                        df = pd.read_excel(BytesIO(file_content))
                        quarterly_data.append(df)
                yearly_data.append(quarterly_data)
            
            if yearly_data:
                combined_dfs = [pd.concat(year_data, ignore_index=True) for year_data in yearly_data]
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                st.write(combined_df)  # Display the combined DataFrame

                if metric == "CRE Concentration":
                    plot_metric_trend(combined_df, "CRE Concentration Ratio (%)", "CRE Concentration Ratio (%)")
                    plot_vertical_bar_chart_common(combined_df, "Quarter", "CRE Concentration Ratio (%)", "CRE Concentration Ratio (%)")
                elif metric == "Loan-to-Deposit Ratio":
                    plot_metric_trend(combined_df, "Loan-to-Deposit Ratio (%)", "Loan-to-Deposit Ratio (%)")
                    plot_vertical_bar_chart_common(combined_df, "Quarter", "Loan-to-Deposit Ratio (%)", "Loan-to-Deposit Ratio (%)")
                elif metric == "Net Interest Margin":
                    plot_metric_trend(combined_df, "Net Interest Margin (%)", "Net Interest Margin (%)")
                    plot_vertical_bar_chart_common(combined_df, "Quarter", "Net Interest Margin (%)", "Net Interest Margin (%)")
            else:
                st.error("No files found for the selected metric")

            # Yearly
            yearly_data = []
            for year_folder in list_folders_in_s3_new(metrics_summary_s3_bucket, f'Yearly/{metric}/'):
                file_key = f"Yearly/{metric}/{year_folder}/{metric}.xlsx"
                file_content = download_file_from_s3(metrics_summary_s3_bucket, file_key)
                if file_content:
                    df = pd.read_excel(BytesIO(file_content))
                    yearly_data.append(df)

            if yearly_data:
                combined_df = pd.concat(yearly_data, ignore_index=True)
                st.write(combined_df)  # Display the combined DataFrame

                if metric == "CRE Concentration":
                    plot_vertical_bar_chart_common(combined_df, "Year", "CRE Concentration Ratio (%)", "CRE Concentration Ratio (%)")
                elif metric == "Loan-to-Deposit Ratio":
                    plot_vertical_bar_chart_common(combined_df, "Year", "Loan-to-Deposit Ratio (%)", "Loan-to-Deposit Ratio (%)")
                elif metric == "Net Interest Margin":
                    plot_vertical_bar_chart_common(combined_df, "Year", "Net Interest Margin (%)", "Net Interest Margin (%)")
            else:
                st.error("No files found for the selected metric")
        else:
            st.error("Please select a metric to generate the graph")
        generate_llm_output(metric)

    st.write("We will have a prompt template in the background to compare the selected metric between the base bank's metric with its peer banks")



def generate_llm_output(metric):
    if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
        with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
            st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache
    
    #input_text = st.text_area("Input text", label_visibility="collapsed") #display a multiline text box with no label
    prompt = """
    Assume you are a seasoned banking analyst and as part of peer bank risk analytics, you need to analyze 
    how the base bank is performing on the metric "{}" against its two peer banks 
    across all the 4 quarters.  Generate a nicely readable summary to clearly show the variance of the base
    as compared to its peer banks to make better business decisions. Give more reasoning of if base bank 
    is performing better or not as compared to its peers on this ratio.
    """.format(metric)
    
    #go_button = st.button("Generate Summary", type="primary") #display a primary button
    
    # if st.button("Generate First Output"): #code in this if block will be run when the button is clicked
        
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = glib.get_rag_response(index=st.session_state.vector_index, question=prompt) #call the model through the supporting library
        
        st.write(response_content) #display the response content
        

# Function for Analyze Metrics - Details (Chatbot)
def analyze_metrics_details():
    # Bank selection
    bank_selection = st.radio("Select a Bank", ['Base Bank', 'Peer Bank'])
    peer_bank = st.selectbox("Select a Peer", ['Bank A', 'Bank B', 'Bank C'], key="peer_bank_selection")

    # Metric selection
    metric = st.selectbox("Select a Metric (optional)", ['Metric 1', 'Metric 2', 'Metric 3'], key="metric_selection")

    # Chat history
    st.write("Chat History")
    st.text_area("", "Ques 1: abc\nAns 1: xyz\nQues 2: abc\nAns 2: xyz\nQues 3: abc\nAns 3: xyz", height=200)

    # Q&A
    user_input = st.text_input("Q & A", "<type a question here & press enter>")
    if st.button("Go"):
        st.write("You asked: ", user_input)
        st.write("Answer: [Processing your question...]")

# Main menu with section IDs
page = st.sidebar.radio("Go to", ["Financial Reports Hub", "Analyze Metrics - Summary", "Analyze Metrics - Details"])

if page == "Financial Reports Hub":
    financial_reports_hub()
elif page == "Analyze Metrics - Summary":
    analyze_metrics_summary()
elif page == "Analyze Metrics - Details":
    analyze_metrics_details()
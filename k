import faiss
import numpy as np
import pandas as pd

import os
import numpy as np
import pandas as pd
import faiss
import json
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
api_key = os.environ["OPEN_AI_KEY"]
azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
api_version = '2023-03-15-preview'

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint="https://llm-api-gateway-v2.azurewebsites.net/",
    model_name = "gpt4o",
    deployment_name = "aiml-interns-gpt-4-turbo-with-vision-model",
    api_version="2023-03-15-preview"
)

#Text Embedding 3 Large
azure_endpoint = "https://apflowllm.openai.azure.com/"  
api_key = os.environ["EMBEDD_KEY"] 
deployment_name = "text-embedding-3-large"  
api_version = "2023-05-15"  

# Step 2: Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    model=deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version
)

def load_invoice_data(row): 
    """
    Create a text representation of an invoice that emphasizes unique aspects.
    
    """
    return (f"Invoice ID: {row['CoraAP__Invoice_No__c']} ### "
            f"Vendor Code: {row['CoraAP__Vendor__c']} ### "
            f"Payment Amount: {row['CoraAP__Total_Amount__c']:.2f} ### "
            f"Issued Date: {row['CoraAP__Invoice_Date__c']}")

def parse_invoice_text(text):
    """
    Parse stored invoice text back into structured format.
  
    """
    parts = text.split(" ### ")
    invoice_no = parts[0].split(": ")[1].strip()
    vendor = parts[1].split(": ")[1].strip()
    amount = float(parts[2].split(": ")[1].strip())
    date = parts[3].split(": ")[1].strip()
    return {"invoice_no": invoice_no, "vendor": vendor, "amount": amount, "date": date}

def calculate_field_similarity(new_inv, stored_inv):
    """
    Calculate field-by-field similarity between two invoices.
    
    """
    # Different weights for different fields
    invoice_weight = 0.25  # Highest weight for invoice number
    amount_weight = 0.25   # High weight for amount
    date_weight = 0.25     # Medium weight for date
    vendor_weight = 0.25   # Lower weight for vendor
    
    # Calculate individual field scores
    invoice_match = 1.0 if new_inv["invoice_no"] == stored_inv["invoice_no"] else 0.0
    vendor_match = 1.0 if new_inv["vendor"] == stored_inv["vendor"] else 0.0
    date_match = 1.0 if new_inv["date"] == stored_inv["date"] else 0.0
    amount_match = 1.0 if new_inv["amount"] == stored_inv["amount"] else 0.0
        
    # Weighted combination
    field_similarity = (
        invoice_weight * invoice_match +
        amount_weight * amount_match +
        date_weight * date_match +
        vendor_weight * vendor_match
    )
    
    return {
        "overall": field_similarity,
        "invoice_match": invoice_match,
        "vendor_match": vendor_match,
        "amount_match": amount_match,
        "date_match": date_match,
    }

def analyze_invoice_with_llm(invoice_analysis):
    """
    Sends the formatted invoice analysis to LLM for reasoning.
    
    Args:
        invoice_analysis: Formatted invoice analysis
        llm: Language model instance
    
    Returns:
        str: LLM's response
    """
    prompt = prompt_template.format(invoice_analysis=invoice_analysis)
    response = llm([SystemMessage(content=prompt)])
    return response.content

# Define Prompt Template for LLM
prompt_template = PromptTemplate(
    input_variables=["invoice_analysis"],
    template="""You are a financial AI assistant specializing in invoice fraud detection.
Your goal is to analyze the provided new invoice with potential duplicates invoices data based on the each rules provided below and make a final decision. If any of the rules matches the criteria then mark the invoice fraudulent.

### ** Some Duplicate Invoice Detection Rules:**
1.Same Invoice Number & Vendor - Check if identical invoice numbers appear from the same vendor, suggesting it might be duplicate.
2.Same Amount & Date from Same Vendor - Look for invoices with matching amounts and dates from the same vendor, indicating possible duplicate processing.
3.Similar Invoice Numbers - Identify invoice numbers with minor variations (INV-1234/INV1234) but matching vendor and amount, suggesting data entry errors. Indicating it is duplicate
4.Matching PO Number & Amount - Flag when multiple invoices reference the same purchase order with identical amounts.
5.Same Invoice Details with Different Vendors - Watch for similar invoice details appearing under different vendor names.Indicating it is duplicate
6.Close Dates & Similar Amounts - Be alert to invoices from the same vendor with overlapping dates and similar amounts.
7.Identical Amounts on Different Dates - Notice when the same exact amount appears on different dates from the same vendor might be duplicate
8.Round Amount Patterns - Be suspicious of large, rounded amounts (like $5000.00) appearing multiple times.
9. Currency Variations - Check when the same invoice number appears with different currencies.
10.Format Variations - Identify when invoice numbers differ only by special characters or formatting (INV#1234/INV_1234).


### **Invoice Similarity Analysis:**  
{invoice_analysis}

### **Provide the following:**
**Final decision:** Is the new invoice a duplicate, likely fraudulent, or distinct?  
**Risk level:** (Low / Medium / High) based on the given details
**Supporting factors:** What key data points support this conclusion?  
**Recommended action:** Should this invoice be flagged for further review?  

Provide a structured response.
"""
)


# Check if index already exists
index_file = "faiss_invoice_index.bin"

if not os.path.exists(index_file):
    # Only build the index if it doesn't exist yet
    print("Building new FAISS index...")
    
    # Load and prepare data
    df = pd.read_csv("invoice_data_c.csv")
    df['invoice_text'] = df.apply(load_invoice_data, axis=1)
    
    # Convert invoice texts into embeddings
    old_invoice_embeddings = embeddings.embed_documents(df['invoice_text'].tolist())
    embedding_dim = len(old_invoice_embeddings[0])
    
    # Use cosine similarity with normalization
    old_invoice_embeddings = np.array(old_invoice_embeddings).astype('float32')
    faiss.normalize_L2(old_invoice_embeddings)
    
    # Create an IndexIDMap to map vectors to their row IDs in the dataframe
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    id_map = faiss.IndexIDMap(index)
    
    # Add vectors with IDs (0, 1, 2, ...) that correspond to dataframe row indices
    ids = np.array(range(len(old_invoice_embeddings))).astype('int64')
    id_map.add_with_ids(old_invoice_embeddings, ids)
    
    # Save index to disk
    faiss.write_index(id_map, index_file)
    print(f"Index saved to {index_file}")
else:
    print(f"Using existing index from {index_file}")

# For searching, load the existing index
id_map = faiss.read_index(index_file)

# Load original data for reference (only needed during search, not for building index)
df = pd.read_csv("invoice_data_c.csv")

# New invoices data and search logic
new_invoices_data = [
    {"invoice_no": "I_57399567", "vendor": "B2", "amount": 810.47, "date": "2024-09-18"},
]

# Format for embedding
new_invoices_text = [
    (f"Invoice ID: {inv['invoice_no']} ### "
     f"Vendor Code: {inv['vendor']} ### "
     f"Payment Amount: {inv['amount']:.2f} ### "
     f"Issued Date: {inv['date']}")
    for inv in new_invoices_data
]

# Embed new invoices
new_invoice_embeddings = embeddings.embed_documents(new_invoices_text)
new_invoice_embeddings = np.array(new_invoice_embeddings).astype('float32')
faiss.normalize_L2(new_invoice_embeddings)

# Search parameters
# Search for similar invoices
for i, new_inv in enumerate(new_invoices_data):
    query_embedding = new_invoice_embeddings[i].reshape(1, -1)
    
    # Search in FAISS index
    similarities, indices = id_map.search(query_embedding, top_k)
    
    # Format new invoice for display
    new_invoice_formatted = {
        "invoice_no": new_inv['invoice_no'],
        "vendor": new_inv['vendor'],
        "amount": new_inv['amount'],
        "date": new_inv['date']
    }
    
    matches = []  # Store matches for this new invoice
    
    # Check matches against threshold
    for j in range(min(top_k, len(indices[0]))):
        match_index = indices[0][j]
        similarity = similarities[0][j]

        matched_row = df.iloc[match_index]
            
            # Extract structured data from the matched invoice
        stored_invoice = {
            "invoice_no": matched_row["CoraAP__Invoice_No__c"],
            "vendor": matched_row["CoraAP__Vendor__c"],
            "amount": matched_row["CoraAP__Total_Amount__c"],
             "date": matched_row["CoraAP__Invoice_Date__c"]
        }

        sim_scores = calculate_field_similarity(new_invoice_formatted, stored_invoice)
        
        if sim_scores["overall"] >= llm_analysis_threshold:
            # Get original data directly from the dataframe using the index
            
            
            matches.append({
                "stored_invoice": stored_invoice,
                "similarity": float(similarity)
            })
    
    if matches:
        # Construct formatted invoice analysis
        invoice_analysis = f"""
        **New Invoice:**
        - Invoice Number: {new_invoice_formatted['invoice_no']}
        - Vendor: {new_invoice_formatted['vendor']}
        - Amount: {new_invoice_formatted['amount']}
        - Date: {new_invoice_formatted['date']}

        **Potential Matches:**
        """
        
        for match in matches:
            stored = match["stored_invoice"]
            invoice_analysis += f"""
            - **Invoice Number:** {stored['invoice_no']}
              **Vendor:** {stored['vendor']}
              **Amount:** {stored['amount']}
              **Date:** {stored['date']}
            """
        
        # Print which invoices are being sent to the prompt
        print("\n⚡ **Sending the following invoice data to LLM:** ⚡")
        print(invoice_analysis)
        
        # Call the LLM for reasoning
        llm_response = analyze_invoice_with_llm(invoice_analysis)
        print(f"\n🔍 **LLM Final Analysis for Invoice {new_invoice_formatted['invoice_no']}:** 🔍\n")
        print(llm_response)
    else:
        print("#################")
        print(f"❌ No potential matches found for Invoice {new_invoice_formatted['invoice_no']}")

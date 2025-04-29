import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os

class AssociationMiner:
    """
    Association rule mining for finding patterns in accident data.
    """
    
    def __init__(self):
        """Initialize the association rule miner."""
        self.frequent_itemsets = None
        self.rules = None
        self.encoder = None
        self.categorical_cols = None
    
    def fit(self, df):
        """
        Prepare the association rule miner (no actual training).
        
        Args:
            df: DataFrame containing railway accident data
        """
        # Identify categorical columns that could be used for association mining
        categorical_cols = [
            'Accident_Type', 'Cause', 'State/Region', 'Train_Involved', 
            'Decade', 'Severity'
        ]
        
        # Filter to columns that exist in the DataFrame
        self.categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        return self
    
    def mine_rules(self, df, selected_fields, min_support=0.05, min_confidence=0.7, min_lift=1.2):
        """
        Mine association rules from the data.
        
        Args:
            df: DataFrame containing railway accident data
            selected_fields: List of fields to use for mining
            min_support: Minimum support for itemsets
            min_confidence: Minimum confidence for rules
            min_lift: Minimum lift for rules
            
        Returns:
            DataFrame with association rules
        """
        # Make sure all selected fields exist in the DataFrame
        for field in selected_fields:
            if field not in df.columns:
                raise ValueError(f"Selected field '{field}' not found in DataFrame")
        
        # Prepare transaction data
        transactions = []
        
        for _, row in df.iterrows():
            # Create items in format "field=value"
            items = []
            for field in selected_fields:
                if pd.notna(row[field]):
                    items.append(f"{field}={row[field]}")
            
            if items:
                transactions.append(items)
        
        # Convert to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        one_hot = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Store encoder
        self.encoder = te
        
        # Generate frequent itemsets
        try:
            self.frequent_itemsets = apriori(
                one_hot, 
                min_support=min_support,
                use_colnames=True
            )
            
            if self.frequent_itemsets.empty:
                return pd.DataFrame()
            
            # Generate association rules
            self.rules = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            
            # Filter by lift
            self.rules = self.rules[self.rules['lift'] >= min_lift]
            
            # Sort by lift
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            return self.rules
            
        except Exception as e:
            print(f"Error in mining association rules: {e}")
            return pd.DataFrame()
    
    def get_rules(self):
        """
        Get the mined association rules.
        
        Returns:
            DataFrame with association rules
        """
        if self.rules is None:
            raise ValueError("No rules have been mined yet. Call 'mine_rules' first.")
        
        return self.rules
    
    def get_frequent_itemsets(self):
        """
        Get the frequent itemsets.
        
        Returns:
            DataFrame with frequent itemsets
        """
        if self.frequent_itemsets is None:
            raise ValueError("No frequent itemsets have been generated yet. Call 'mine_rules' first.")
        
        return self.frequent_itemsets
    
    def save(self, path):
        """
        Save the mined rules to disk.
        
        Args:
            path: Path to save the rules
        """
        if self.rules is None:
            raise ValueError("No rules have been mined yet. Call 'mine_rules' first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save rules and itemsets
        joblib.dump({
            'rules': self.rules,
            'frequent_itemsets': self.frequent_itemsets,
            'encoder': self.encoder,
            'categorical_cols': self.categorical_cols
        }, path)
    
    def load(self, path):
        """
        Load the mined rules from disk.
        
        Args:
            path: Path to the saved rules
        """
        if not os.path.exists(path):
            raise ValueError(f"Rules file '{path}' not found")
        
        # Load rules and itemsets
        saved_data = joblib.load(path)
        
        self.rules = saved_data['rules']
        self.frequent_itemsets = saved_data['frequent_itemsets']
        self.encoder = saved_data['encoder']
        self.categorical_cols = saved_data['categorical_cols']
        
        return self

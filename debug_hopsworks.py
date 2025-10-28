#!/usr/bin/env python3
"""
Debug Hopsworks Feature Group Retrieval
=======================================

This script helps debug the issue with retrieving features from Hopsworks.
"""

import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def debug_feature_group_retrieval():
    """Debug the feature group retrieval process."""
    
    try:
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            project=os.getenv('HOPSWORKS_PROJECT'),
            api_key_value=os.getenv('HOPSWORKS_API_KEY')
        )
        fs = project.get_feature_store()
        print("✅ Connected successfully")
        
        print("\n🔍 Testing different versions...")
        
        # Test different versions to see which ones exist
        for version in [1, 2, 3, 4, 5]:
            try:
                feature_group = fs.get_feature_group(
                    name='karachifeatures',
                    version=version
                )
                
                if feature_group is not None:
                    print(f"✅ Found karachifeatures v{version}")
                    print(f"   Name: {getattr(feature_group, 'name', 'MISSING')}")
                    print(f"   Version: {getattr(feature_group, 'version', 'MISSING')}")
                    
                    # Try to get some basic info
                    try:
                        query = feature_group.select_all()
                        df = query.read()  # Remove limit parameter that's not supported
                        print(f"   Records available: Yes (sample shape: {df.shape})")
                        return feature_group, version  # Return the working one
                    except Exception as e:
                        print(f"   Records available: No ({e})")
                else:
                    print(f"❌ karachifeatures v{version} returned None")
                    
            except Exception as e:
                print(f"❌ karachifeatures v{version} error: {e}")
        
        print("\n💡 DIAGNOSIS:")
        print("   The feature group exists but may be empty or corrupted.")
        print("   You need to run the pipeline to populate it with data.")
        print("   Run: python unified_aqi_hopsworks_pipeline.py backfill")
        
        return None, None
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    debug_feature_group_retrieval()
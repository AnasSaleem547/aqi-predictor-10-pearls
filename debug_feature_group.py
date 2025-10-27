#!/usr/bin/env python3
"""
Debug script to test Hopsworks feature group retrieval issue.
"""

import os
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT")
FEATURE_GROUP_NAME = "karachifeatures"
FEATURE_GROUP_VERSION = 1

def debug_feature_group():
    """Debug the feature group retrieval issue."""
    print("🔧 DEBUGGING FEATURE GROUP ISSUE")
    print("=" * 50)
    
    try:
        # Connect to Hopsworks
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        print(f"✅ Connected to project: {project.name}")
        
        # Get feature store
        fs = project.get_feature_store()
        print(f"✅ Got feature store: {fs.name}")
        
        # Try to get the feature group
        print(f"\n🔍 Attempting to get feature group: {FEATURE_GROUP_NAME}")
        try:
            fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            print(f"✅ Retrieved feature group: {fg}")
            print(f"   Type: {type(fg)}")
            print(f"   Name: {fg.name if hasattr(fg, 'name') else 'No name attribute'}")
            print(f"   Version: {fg.version if hasattr(fg, 'version') else 'No version attribute'}")
            
            # Test if we can call insert method
            print(f"\n🧪 Testing feature group methods...")
            print(f"   Has insert method: {hasattr(fg, 'insert')}")
            print(f"   Has read method: {hasattr(fg, 'read')}")
            
            # Try to read some data to test if it's working
            try:
                print(f"\n📖 Attempting to read data...")
                data = fg.read(limit=1)
                print(f"✅ Successfully read data: {len(data)} records")
                print(f"   Columns: {list(data.columns)}")
            except Exception as read_error:
                print(f"❌ Failed to read data: {str(read_error)}")
            
            return fg
            
        except Exception as get_error:
            print(f"❌ Failed to get feature group: {str(get_error)}")
            print(f"   Error type: {type(get_error)}")
            return None
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return None

if __name__ == "__main__":
    debug_feature_group()
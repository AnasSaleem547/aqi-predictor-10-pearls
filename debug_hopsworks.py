#!/usr/bin/env python3
"""
Debug script to check Hopsworks feature groups and understand connection issues.
"""

import os
from dotenv import load_dotenv
import hopsworks

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT")
FEATURE_GROUP_NAME = "karachifeatures"
FEATURE_GROUP_VERSION = 1

def debug_hopsworks():
    """Debug Hopsworks connection and feature groups."""
    try:
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        fs = project.get_feature_store()
        print("✅ Connected to Hopsworks successfully")
        
        # List all feature groups
        print("\n📋 Listing all feature groups:")
        try:
            feature_groups = fs.get_feature_groups()
            for fg in feature_groups:
                print(f"  - {fg.name} (version {fg.version})")
        except Exception as e:
            print(f"❌ Error listing feature groups: {str(e)}")
        
        # Try to get our specific feature group
        print(f"\n🔍 Checking for feature group: {FEATURE_GROUP_NAME}")
        try:
            fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            if fg:
                print(f"✅ Found feature group: {fg.name}")
                print(f"   Version: {fg.version}")
                print(f"   Description: {fg.description}")
                print(f"   Primary key: {fg.primary_key}")
                print(f"   Event time: {fg.event_time}")
                
                # Try to get schema
                try:
                    schema = fg.get_feature_descriptions()
                    print(f"   Schema: {schema}")
                except Exception as schema_error:
                    print(f"   Schema error: {str(schema_error)}")
                    
            else:
                print("❌ Feature group returned None")
        except Exception as e:
            print(f"❌ Error getting feature group: {str(e)}")
            
            # Try to create a new one
            print(f"\n🆕 Attempting to create feature group: {FEATURE_GROUP_NAME}")
            try:
                new_fg = fs.create_feature_group(
                    name=FEATURE_GROUP_NAME,
                    version=FEATURE_GROUP_VERSION,
                    description="Karachi AQI features with EPA calculations and engineered features",
                    primary_key=["datetime"],
                    event_time="datetime",
                    online_enabled=True,
                )
                print(f"✅ Created feature group: {new_fg.name}")
            except Exception as create_error:
                print(f"❌ Failed to create feature group: {str(create_error)}")
        
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {str(e)}")

if __name__ == "__main__":
    debug_hopsworks()
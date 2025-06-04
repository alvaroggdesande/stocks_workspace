# minimal_obb_test.py (place this in your project_root/data/ directory for consistent import paths)
import os
print("--- Minimal OBB Test ---")

# Attempt to load from config.py first
import sys
import os
# Add the parent directory (project root) to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
FMP_API_KEY_FROM_CONFIG = None
try:
    from config.config import FMP_API_KEY
    if FMP_API_KEY and not FMP_API_KEY.startswith("YOUR_"):
        FMP_API_KEY_FROM_CONFIG = FMP_API_KEY
        print(f"Loaded FMP_API_KEY from config: {FMP_API_KEY_FROM_CONFIG[:4]}...")
except ImportError:
    print("config.py not found or FMP_API_KEY not in it.")

# Ensure the environment variable is set regardless, as a primary attempt
if FMP_API_KEY_FROM_CONFIG:
    os.environ['OPENBB_FMP_API_KEY'] = FMP_API_KEY_FROM_CONFIG
    print(f"Set os.environ['OPENBB_FMP_API_KEY'] using key from config.py.")
else:
    print("Did not set os.environ['OPENBB_FMP_API_KEY'] from config.py (key not found or placeholder).")

print(f"Value of OPENBB_FMP_API_KEY in current environment: {os.getenv('OPENBB_FMP_API_KEY')}")

from openbb import obb

# Attempt 1: Rely on environment variable (as before)
print("\nAttempt 1: Relying on environment variable for FMP...")
try:
    ratios_obj_env = obb.equity.fundamental.ratios(
        symbol="AAPL",
        period="annual",
        limit=1,
        provider="fmp"
    )
    if ratios_obj_env and hasattr(ratios_obj_env, 'results') and ratios_obj_env.results:
        print("SUCCESS (Attempt 1): Fetched ratios using environment variable.")
        print(ratios_obj_env.to_df())
    else:
        print("FAILURE (Attempt 1): FMP call with env var did not yield results or failed silently.")
except Exception as e_env:
    print(f"ERROR (Attempt 1) during obb.equity.fundamental.ratios call with env var: {e_env}")


# Attempt 2: Try to pass credentials directly if FMP_API_KEY_FROM_CONFIG is available
# This syntax for passing credentials directly has changed in OpenBB v4.
# The new way is often to set it via obb.user.credentials or when fetching a provider specific object.
# Let's try the documented way for OpenBB v4 to set user-level credentials for a provider.
# This is more likely to work than passing to the function directly.
if FMP_API_KEY_FROM_CONFIG:
    print("\nAttempt 2: Trying to set FMP credentials via obb.user.credentials.fmp_api_key...")
    try:
        # This is the documented way to set credentials for a provider in OpenBB v4+
        # It saves it to the user's local settings for the SDK.
        obb.user.credentials.fmp_api_key = FMP_API_KEY_FROM_CONFIG
        print(f"Set obb.user.credentials.fmp_api_key.")

        ratios_obj_creds = obb.equity.fundamental.ratios(
            symbol="AAPL",
            period="annual",
            limit=1,
            provider="fmp" # This should now use the credential set above
        )
        if ratios_obj_creds and hasattr(ratios_obj_creds, 'results') and ratios_obj_creds.results:
            print("SUCCESS (Attempt 2): Fetched ratios after setting obb.user.credentials.")
            print(ratios_obj_creds.to_df())
        else:
            print("FAILURE (Attempt 2): FMP call after setting obb.user.credentials did not yield results or failed silently.")
    except AttributeError as ae:
        print(f"AttributeError (Attempt 2): 'obb.user.credentials' does not have 'fmp_api_key' or 'user' has no 'credentials'. Error: {ae}")
        print("This suggests the path to set credentials programmatically might be different for your SDK version or FMP.")
    except Exception as e_creds:
        print(f"ERROR (Attempt 2) during obb.equity.fundamental.ratios call with direct creds: {e_creds}")
else:
    print("\nSkipping Attempt 2 (direct credentials) as FMP_API_KEY_FROM_CONFIG is not available.")


print("--- Test Complete ---")
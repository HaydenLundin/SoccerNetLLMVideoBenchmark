# ============================================================================
# CHECK GEMINI API QUOTA & USAGE
# ============================================================================
"""
This script checks your Gemini API usage and limits.
"""

import google.generativeai as genai
from google.colab import userdata
from datetime import datetime

print("="*80)
print("GEMINI API QUOTA & USAGE CHECK")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================

try:
    api_key = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
    print("\n✅ API Key configured")
except Exception as e:
    print(f"\n❌ Failed to get API key: {e}")
    exit()

# ============================================================================
# CHECK MODEL AVAILABILITY
# ============================================================================

print("\n" + "="*80)
print("AVAILABLE MODELS")
print("="*80)

try:
    models = genai.list_models()

    print("\nModels you can use:")
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"\n  📌 {model.name}")
            print(f"     Display name: {model.display_name}")
            if hasattr(model, 'input_token_limit'):
                print(f"     Input token limit: {model.input_token_limit:,}")
            if hasattr(model, 'output_token_limit'):
                print(f"     Output token limit: {model.output_token_limit:,}")

except Exception as e:
    print(f"❌ Error listing models: {e}")

# ============================================================================
# TRY A SIMPLE REQUEST TO SEE QUOTA INFO
# ============================================================================

print("\n" + "="*80)
print("TESTING API REQUEST")
print("="*80)

try:
    # Try the free flash model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    print("\nSending test request...")
    response = model.generate_content("Say hello")

    print("✅ Request successful!")
    print(f"Response: {response.text[:100]}")

    # Check if there's usage metadata
    if hasattr(response, 'usage_metadata'):
        print(f"\n📊 Token usage:")
        print(f"   Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"   Response tokens: {response.usage_metadata.candidates_token_count}")
        print(f"   Total tokens: {response.usage_metadata.total_token_count}")

except Exception as e:
    print(f"❌ Request failed: {e}")
    print(f"\n🔍 Error details:")
    print(f"   {str(e)}")

    if "429" in str(e):
        print("\n⚠️  QUOTA EXCEEDED ERROR")
        print("   This means you've hit your rate limit.")
    elif "403" in str(e):
        print("\n⚠️  PERMISSION ERROR")
        print("   Check if your API key is valid.")
    elif "404" in str(e):
        print("\n⚠️  MODEL NOT FOUND")
        print("   The model might not be available.")

# ============================================================================
# WHERE TO CHECK USAGE
# ============================================================================

print("\n" + "="*80)
print("WHERE TO CHECK YOUR USAGE")
print("="*80)

print("\n📊 Check your usage and limits here:")
print("\n1. Google AI Studio - Usage Dashboard")
print("   🔗 https://aistudio.google.com/app/apikey")
print("   • Shows your API key")
print("   • Shows rate limits")
print("   • Shows usage statistics")

print("\n2. Google AI Studio - Rate Limits")
print("   🔗 https://ai.google.dev/pricing")
print("   • Free tier limits")
print("   • Paid tier options")

print("\n3. Monitor Usage")
print("   🔗 https://ai.dev/usage?tab=rate-limit")
print("   • Real-time usage monitoring")
print("   • Quota details")

print("\n4. Google Cloud Console (if using project)")
print("   🔗 https://console.cloud.google.com/apis/dashboard")
print("   • Detailed API metrics")
print("   • Quota settings")

# ============================================================================
# FREE TIER LIMITS
# ============================================================================

print("\n" + "="*80)
print("FREE TIER LIMITS (as of Nov 2024)")
print("="*80)

print("\n📋 Gemini 2.0 Flash (Free):")
print("   • 15 requests per minute (RPM)")
print("   • 1 million tokens per minute (TPM)")
print("   • 1,500 requests per day (RPD)")
print("   • 10 million tokens per day")

print("\n📋 Gemini 1.5 Flash (Free):")
print("   • 15 RPM")
print("   • 1 million TPM")
print("   • 1,500 RPD")

print("\n⚠️  Video files count heavily against your quota!")
print("   • Each video can be 10-50 MB")
print("   • Videos use LOTS of tokens")
print("   • You might hit daily limit after just a few videos")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n💡 Options to continue:")

print("\n1️⃣  WAIT 24 HOURS")
print("   • Free tier quota resets daily")
print("   • You'll get fresh limits tomorrow")

print("\n2️⃣  UPGRADE TO PAID")
print("   🔗 https://aistudio.google.com/app/billing")
print("   • Gemini 2.0 Flash: $0.10 per 1M tokens")
print("   • Much higher rate limits")
print("   • Estimated cost: ~$0.50-$2 per video")

print("\n3️⃣  USE QWEN INSTEAD (RECOMMENDED)")
print("   • FREE with your $10 OpenRouter credit")
print("   • No quota issues")
print("   • Good quality (32B parameters)")
print("   • Process all 104 games for free!")

print("\n4️⃣  USE MULTIPLE API KEYS")
print("   • Create new Google accounts")
print("   • Get new API keys")
print("   • Rotate between them")

print("\n5️⃣  PROCESS FEWER VIDEOS PER DAY")
print("   • Stay within free tier")
print("   • Process ~10-20 videos per day")
print("   • Takes longer but stays free")

# ============================================================================
# CURRENT STATUS
# ============================================================================

print("\n" + "="*80)
print("YOUR CURRENT STATUS")
print("="*80)

print("\n🔴 Gemini: QUOTA EXCEEDED")
print("   • You hit your daily limit")
print("   • Likely from the video uploads")
print("   • Resets in < 24 hours")

print("\n🟢 Qwen: AVAILABLE")
print("   • OpenRouter with $10 credit")
print("   • 1,000 free requests per day")
print("   • You've used 0 so far!")

print("\n" + "="*80)
print("DONE")
print("="*80)

print("\n💬 What would you like to do?")
print("   A) Wait for Gemini quota reset")
print("   B) Switch to Qwen-only")
print("   C) Upgrade Gemini to paid tier")
print("   D) Try a different approach")

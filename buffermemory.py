# Install LangChain first if you haven't
# pip install langchain langchain-community langchain-openai
import json
from langchain.memory import ConversationSummaryBufferMemory, CombinedMemory, SimpleMemory
from langchain_community.llms.fake import FakeListLLM  # Using fake LLM for demo

# 1. Create a language model instance (using fake LLM for demo - no API key needed)
responses = [
    "Hello! I'm doing well, thank you for asking.",
    "Sure! Why don't scientists trust atoms? Because they make up everything!",
    "I don't have real-time weather data, but I hope it's nice where you are!",
    "You're welcome! Have a great day!"
]
llm = FakeListLLM(responses=responses)

# 2. Initialize conversation memory
user_profile = {"name": "Dhruv", "plan": "Pro", "preferences": ["jokes", "weather"]}
profile_memory = SimpleMemory(memories={"user_profile": json.dumps(user_profile)})
summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=50, return_messages=True)
memory = CombinedMemory(memories=[profile_memory, summary_memory])

# 3. Simulate a conversation
conversation_inputs = [
    "Hello, how are you?",
    "Can you tell me a joke?",
    "What's the weather like today?",
    "Thanks, goodbye!"
]

for user_input in conversation_inputs:
    # Update memory with user input and a dummy assistant response
    assistant_response = llm.invoke(f"Respond to: {user_input}")
    memory.save_context({"input": user_input}, {"output": assistant_response})
# 4. Inspect memory content in detail
print("=" * 60)
print("🔍 COMPLETE LANGCHAIN MEMORY CONTENTS")
print("=" * 60)

# Get all memory variables
memory_vars = memory.load_memory_variables({})
print("\n📋 MEMORY VARIABLES:")
print(f"Keys available: {list(memory_vars.keys())}")
for key, value in memory_vars.items():
    print(f"\n{key}:")
    if isinstance(value, list):
        for i, item in enumerate(value):
            print(f"  [{i}] {item}")
    else:
        print(f"  {value}")

# Show chat memory details (from the summary memory)
print("\n💬 CHAT MEMORY DETAILS (from summary_memory):")
print(f"Chat memory type: {type(summary_memory.chat_memory)}")
print(f"Chat memory: {summary_memory.chat_memory}")

# Show all messages in chat memory
if hasattr(summary_memory.chat_memory, 'messages'):
    print(f"\n📝 ALL MESSAGES IN CHAT MEMORY ({len(summary_memory.chat_memory.messages)} messages):")
    for i, msg in enumerate(summary_memory.chat_memory.messages):
        print(f"  [{i}] Type: {type(msg).__name__}")
        print(f"      Content: {msg.content}")
        if hasattr(msg, 'type'):
            print(f"      Role: {msg.type}")
        if hasattr(msg, 'additional_kwargs'):
            print(f"      Additional: {msg.additional_kwargs}")
        print()

# Show summary buffer
print("\n📊 SUMMARY BUFFER (from summary_memory):")
print(f"Summary: {summary_memory.moving_summary_buffer}")
print(f"Summary type: {type(summary_memory.moving_summary_buffer)}")

# Show memory configuration
print("\n⚙️ MEMORY CONFIGURATION (summary_memory):")
print(f"Max token limit: {summary_memory.max_token_limit}")
print(f"Return messages: {summary_memory.return_messages}")
print(f"Memory key: {summary_memory.memory_key}")
print(f"Input key: {summary_memory.input_key}")
print(f"Output key: {summary_memory.output_key}")

# Show all attributes of the memory object
print("\n🔧 ALL MEMORY OBJECT ATTRIBUTES:")
for attr in dir(memory):
    if not attr.startswith('_'):
        try:
            value = getattr(memory, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: <Error accessing: {e}>")

# Show the complete memory object as dictionary
print("\n📦 COMPLETE MEMORY OBJECT (as dict):")
try:
    memory_dict = memory.__dict__
    for key, value in memory_dict.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"  Error accessing __dict__: {e}")

print("\n" + "=" * 60)

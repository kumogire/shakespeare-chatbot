"""
Shakespeare Data Preparation Script
Downloads Shakespeare's complete works and prepares them for training.
"""

import os
import sys
import pickle
import requests
import numpy as np
from pathlib import Path

def download_shakespeare():
    """Download Shakespeare's complete works from Project Gutenberg"""
    
    # Create data directory
    data_dir = Path('data/shakespeare')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URL for Shakespeare's complete works
    url = 'https://www.gutenberg.org/files/100/100-0.txt'
    input_file = data_dir / 'input.txt'
    
    print("📚 Downloading Shakespeare's complete works from Project Gutenberg...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw text
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"✅ Downloaded {len(response.text):,} characters to {input_file}")
        return input_file
        
    except Exception as e:
        print(f"❌ Error downloading Shakespeare data: {e}")
        print("📥 Trying alternative method...")
        
        # Fallback: create a sample if download fails
        sample_text = create_sample_shakespeare()
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
            
        print(f"✅ Created sample Shakespeare data: {len(sample_text):,} characters")
        return input_file

def create_sample_shakespeare():
    """Create a sample Shakespeare text for testing"""
    return """
THE COMPLETE WORKS OF WILLIAM SHAKESPEARE

ROMEO AND JULIET

PROLOGUE

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventured piteous overthrows
Do with their death bury their parents' strife.

HAMLET

HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd.

MACBETH

MACBETH:
Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? Or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?

SONNET 18

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course, untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

""" * 100  # Repeat to make it larger

def clean_text(text):
    """Clean and prepare the text for training"""
    
    print("🧹 Cleaning text...")
    
    # Find the start and end of the actual content
    # Remove Project Gutenberg header and footer
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "THE COMPLETE WORKS OF WILLIAM SHAKESPEARE",
        "ROMEO AND JULIET"
    ]
    
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook"
    ]
    
    # Find content boundaries
    content_start = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            content_start = max(content_start, pos)
            break
    
    content_end = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            content_end = pos
            break
    
    # Extract main content
    if content_start > 0:
        text = text[content_start:]
    if content_end < len(text):
        text = text[:content_end]
    
    # Basic cleaning
    # Remove excessive whitespace but preserve structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip very short lines that are likely artifacts
        if len(line) > 0:
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    print(f"📝 Cleaned text: {len(cleaned_text):,} characters")
    return cleaned_text

def prepare_data():
    """Main data preparation function"""
    
    print("🎭 Preparing Shakespeare dataset for training...")
    
    # Download the data
    input_file = download_shakespeare()
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"📖 Read {len(data):,} characters from input file")
    
    # Clean the text
    data = clean_text(data)
    
    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"📚 Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
    
    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the data
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Split into train and validation sets
    data_encoded = encode(data)
    n = len(data_encoded)
    train_data = data_encoded[:int(n*0.9)]
    val_data = data_encoded[int(n*0.9):]
    
    print(f"🚂 Train split: {len(train_data):,} tokens")
    print(f"✅ Validation split: {len(val_data):,} tokens")
    
    # Save the data
    data_dir = Path('data/shakespeare')
    
    # Save as numpy arrays for efficient loading
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    
    train_ids.tofile(data_dir / 'train.bin')
    val_ids.tofile(data_dir / 'val.bin')
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars
    }
    
    with open(data_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"💾 Saved training data to {data_dir}/")
    print(f"   - train.bin: {len(train_data):,} tokens")
    print(f"   - val.bin: {len(val_data):,} tokens") 
    print(f"   - meta.pkl: vocabulary and mappings")
    
    # Print some statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total characters: {len(data):,}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Train/Val split: {len(train_data):,} / {len(val_data):,}")
    print(f"   Data type: {train_ids.dtype}")
    
    # Show a sample
    print("\n📝 Sample text (first 200 characters):")
    print("-" * 50)
    print(data[:200])
    print("-" * 50)
    
    return data_dir

def verify_data():
    """Verify that the prepared data is correct"""
    
    data_dir = Path('data/shakespeare')
    
    try:
        # Load metadata
        with open(data_dir / 'meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        
        # Load binary data
        train_data = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
        val_data = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
        
        print("✅ Data verification successful!")
        print(f"   Vocabulary size: {meta['vocab_size']}")
        print(f"   Train tokens: {len(train_data):,}")
        print(f"   Validation tokens: {len(val_data):,}")
        
        # Test encode/decode
        itos = meta['itos']
        sample_text = ''.join([itos[i] for i in train_data[:100]])
        print(f"\n📖 Sample decoded text:")
        print(sample_text)
        
        return True
        
    except Exception as e:
        print(f"❌ Data verification failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🎭 Shakespeare Dataset Preparation")
    print("=" * 50)
    
    try:
        # Prepare the data
        data_dir = prepare_data()
        
        # Verify the data
        if verify_data():
            print("\n🎉 Data preparation completed successfully!")
            print(f"📁 Data saved to: {data_dir.absolute()}")
            print("\n🚀 Next steps:")
            print("   1. Run 'make train' to train the model")
            print("   2. Or run 'python src/train.py' directly")
        else:
            print("\n❌ Data preparation completed with errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error during data preparation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
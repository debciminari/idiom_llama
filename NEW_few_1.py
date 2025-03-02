import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import re
import numpy as np

# PART 1: LOAD DATA FUNCTIONS

def load_json(filename):
    """Load JSON data from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def load_templates(template_file):
    """Load prompt templates from a JSON file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading templates from {template_file}: {e}")
        return []

def load_demonstrations(demo_file):
    """Load demonstrations from a JSON file."""
    try:
        with open(demo_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading demonstrations from {demo_file}: {e}")
        return []

def extract_language_from_filename(filename):
    """Extract language code from the filename."""
    # Extract language code from filename pattern like 'IT.json', 'EN_test.json', etc.
    match = re.search(r'(?i)([a-z]{2})(_|\.|$)', filename)
    if match:
        return match.group(1).lower()  # Return lowercase language code
    else:
        print(f"Warning: Could not extract language from filename {filename}. Defaulting to 'en'.")
        return 'en'  # Default to English if pattern doesn't match

def select_template(instruction_lang, templates):
    """Select template based on the instruction language."""
    for template in templates:
        if template['lang'].lower() == instruction_lang.lower():
            return template
    print(f"Warning: No template found for language: {instruction_lang}. Using default.")
    # Return a default template if none matches
    return {
        "lang": "en",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_demo": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n"
    }

def select_demonstrations(instruction_lang, demonstrations):
    """
    Select demonstrations matching the instruction language and input language.
    
    Args:
        instruction_lang (str): Language of the instruction (e.g., 'it', 'en', 'pt')
        demonstrations (list): List of demonstration examples
    
    Returns:
        list: Matching demonstrations
    """
    if not demonstrations:
        print("Warning: No demonstrations provided.")
        return []

    matching_demos = []
    
    # Language-specific markers and instruction phrases
    lang_markers = {
        'it': ['frase', 'espressioni', 'costruzioni', 'rilevare', 'individuare'],
        'en': ['sentence', 'idiomatic', 'expressions', 'identify', 'spot'],
        'pt': ['frase', 'expressões', 'construções', 'identificar']
    }
    
    # Look for demonstrations with matching input language
    for demo in demonstrations:
        try:
            # Check if the demonstration has a language-specific input field
            input_key = f"input_{instruction_lang}"
            
            # If the specific input language field exists
            if input_key in demo:
                # Get the instruction as a string
                instruction = str(demo.get('instruction', '')).lower()
                
                # Check for language markers in the instruction
                lang_match = any(
                    marker in instruction 
                    for marker in lang_markers.get(instruction_lang, [])
                )
                
                # If there's a language match, add the demonstration
                if lang_match:
                    matching_demos.append(demo)
        
        except Exception as e:
            print(f"Error processing demonstration: {e}")
            continue
    
    # If no language-specific matches, fall back to all demonstrations with the right input key
    if not matching_demos:
        print(f"Warning: No demonstrations found for language {instruction_lang}. Using alternative approach.")
        
        input_key = f"input_{instruction_lang}"
        matching_demos = [demo for demo in demonstrations if input_key in demo]
    
    # Fallback to any available demonstrations
    if not matching_demos:
        print(f"Warning: No demonstrations found. Using any available demonstrations.")
        matching_demos = demonstrations[:min(3, len(demonstrations))]
    
    # Final sanity check and limit to 3 demonstrations
    matching_demos = matching_demos[:3]
    
    if not matching_demos:
        print("Critical warning: No demonstrations could be selected.")
    
    return matching_demos



# PART 2: PROCESS FILES

def process_file(filename):
    """Process a single JSON file and organize samples by language."""
    json_data = load_json(filename)
    if not json_data:
        return {}
    
    # Determine the instruction language from the filename
    instruction_lang = extract_language_from_filename(filename)
    
    # Separate data by input language (EN, IT, PT)
    separated_data = {
        f"{instruction_lang.upper()}_EN": [],
        f"{instruction_lang.upper()}_IT": [],
        f"{instruction_lang.upper()}_PT": []
    }
    
    # Process each sample
    for sample in json_data:
        # Determine the input language from the keys
        input_lang = None
        for key in sample:
            if key.startswith('input_'):
                input_lang = key.split('_')[1].upper()
                break
        
        # If input language isn't specified in a separate key, try to determine from 'input'
        if not input_lang and 'input' in sample:
            # This is a simplified approach - you might want a more sophisticated language detection
            input_text = sample['input']
            
            # Apply basic heuristics to guess the language
            if any(word in input_text.lower() for word in ['the', 'is', 'and', 'of']):
                input_lang = 'EN'
            elif any(word in input_text.lower() for word in ['il', 'la', 'di', 'e', 'sono']):
                input_lang = 'IT'
            elif any(word in input_text.lower() for word in ['o', 'a', 'de', 'da', 'em']):
                input_lang = 'PT'
            else:
                # Default to same as instruction language if undetermined
                input_lang = instruction_lang.upper()
        
        # Add the sample to the appropriate language combination
        if input_lang:
            key = f"{instruction_lang.upper()}_{input_lang}"
            if key in separated_data:
                separated_data[key].append(sample)
            else:
                # If the key doesn't exist, create it
                separated_data[key] = [sample]
    
    return separated_data

def process_all_files(filenames):
    """Process multiple files and combine their data."""
    all_separated_data = {}
    
    for filename in filenames:
        # Process each file
        separated_data = process_file(filename)
        
        # Add data to the combined dictionary
        for key, samples in separated_data.items():
            if key not in all_separated_data:
                all_separated_data[key] = []
            all_separated_data[key].extend(samples)
    
    return all_separated_data

# PART 3: GENERATE PROMPTS

def create_demonstration_prompts(demonstrations, template):
    """Create prompts for demonstrations using the template."""
    demo_prompts = []
    
    # Get the template format - for demonstrations we use the same format as for main input
    demo_template = template.get("prompt_input", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
    
    # Create a prompt for each demonstration
    for demo in demonstrations:
        instruction = demo.get("instruction", "")
        
        # Get the input from the appropriate field based on language
        input_text = demo.get("input", "")
        if not input_text:
            # Try language-specific input fields
            for lang in ["en", "it", "pt"]:
                input_key = f"input_{lang}"
                if input_key in demo:
                    input_text = demo[input_key]
                    break
        
        output = demo.get("output", "")
        
        # Format the demonstration prompt
        # Append the output to the base template
        demo_prompt = demo_template.format(
            instruction=instruction,
            input=input_text
        )
        
        # Add the response/output
        demo_prompt += f"{output}\n\n"
        
        demo_prompts.append(demo_prompt)
    
    return "".join(demo_prompts)

def generate_full_prompt(sample, templates, demonstrations, filename):
    """Generate a full prompt with demonstrations + main sample."""
    # Extract the instruction language from the filename
    instruction_lang = extract_language_from_filename(filename)
    
    # Select the template based on the instruction language
    template = select_template(instruction_lang, templates)
    
    # Select relevant demonstrations for this language
    relevant_demos = select_demonstrations(instruction_lang, demonstrations)
    
    # Create demonstration prompts
    demo_prompts = create_demonstration_prompts(relevant_demos, template)
    
    # Get instruction and input from the sample
    instruction = sample.get('instruction', '')
    
    # Try to get input from language-specific fields, or fall back to the generic input field
    input_text = ""
    for lang in ["en", "it", "pt"]:
        input_key = f"input_{lang}"
        if input_key in sample:
            input_text = sample[input_key]
            break
    
    # If no language-specific input found, try generic input field
    if not input_text and 'input' in sample:
        input_text = sample['input']
    
    # Generate the main prompt
    prompt_template = template.get("prompt_input", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
    main_prompt = prompt_template.format(instruction=instruction, input=input_text)
    
    # Combine demonstrations and main prompt
    full_prompt = demo_prompts + main_prompt
    
    return full_prompt

def tokenize_prompt(prompt, tokenizer, cutoff_len, add_eos_token=True):
    """Tokenize a prompt for the model."""
    # Make sure we're using the correct return_tensors format
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors="pt"  # This returns PyTorch tensors
    )
    
    # Add EOS token if needed and if there's room
    if add_eos_token and tokenized.input_ids.shape[1] < cutoff_len:
        eos_id = tokenizer.eos_token_id
        # Check if the last token is already the EOS token
        if tokenized.input_ids[0, -1].item() != eos_id:
            # Add EOS token
            new_input_ids = torch.cat([
                tokenized.input_ids, 
                torch.tensor([[eos_id]], device=tokenized.input_ids.device)
            ], dim=1)
            new_attention_mask = torch.cat([
                tokenized.attention_mask,
                torch.tensor([[1]], device=tokenized.attention_mask.device)
            ], dim=1)
            
            tokenized.input_ids = new_input_ids
            tokenized.attention_mask = new_attention_mask
    
    return tokenized


# PART 4: MODEL LOADING AND INFERENCE

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with proper device placement."""
    try:
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_text(prompt, tokenizer, model, device, stop_words=None, max_new_tokens=100):
    """Generate text from a given prompt."""
    if stop_words is None:
        stop_words = ["\n\n", "###"]
    
    try:
        # Tokenize the prompt directly here (don't use the separate function to ensure consistency)
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt"
        ).to(device)
        
        # Record the input length to extract only the new tokens later
        input_length = inputs.input_ids.shape[1]
        
        # Set up stopping criteria
        stop_token_ids = []
        for word in stop_words:
            # Get the token IDs for each stop word/phrase
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                stop_token_ids.append(word_tokens[0])
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.2,
                repetition_penalty=1.2,
                stopping_criteria=stopping_criteria
            )
        
        # Extract only the newly generated text
        generated_text = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text.strip()
    except Exception as e:
        print(f"Error in text generation: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_for_all_samples(all_data, templates, demonstrations, tokenizer, model, device):
    """Generate text for all samples in the dataset."""
    generated_data = {}
    
    # Process each language combination
    for key, samples in all_data.items():
        if not samples:
            continue
        
        print(f"Generating responses for {key} ({len(samples)} samples)...")
        generated_data[key] = []
        
        # Extract instruction language from the key
        instruction_lang = key.split('_')[0].lower()
        filename = f"{instruction_lang}.json"  # Create filename for template selection
        
        # Process each sample
        for i, sample in enumerate(samples):
            try:
                # Generate the full prompt with demonstrations
                full_prompt = generate_full_prompt(
                    sample=sample,
                    templates=templates,
                    demonstrations=demonstrations,
                    filename=filename
                )
                
                # Generate text from the prompt
                generated_text = generate_text(
                    prompt=full_prompt,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
                
                # Get the actual input and output for reference
                input_text = ""
                for lang in ["en", "it", "pt"]:
                    input_key = f"input_{lang}"
                    if input_key in sample:
                        input_text = sample[input_key]
                        break
                if not input_text and 'input' in sample:
                    input_text = sample['input']
                
                # Store the results
                generated_data[key].append({
                    "instruction": sample.get("instruction", ""),
                    "input": input_text,
                    "output": sample.get("output", ""),
                    "generated_text": generated_text or "ERROR: Failed to generate text"  # Provide fallback
                })
                
                # Print progress for long runs
                if (i+1) % 10 == 0 or i == len(samples) - 1:
                    print(f"Processed {i+1}/{len(samples)} samples for {key}")
            
            except Exception as e:
                print(f"Error processing sample {i} in {key}: {e}")
                # Add the sample with error information
                generated_data[key].append({
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "output": sample.get("output", ""),
                    "generated_text": f"ERROR: {str(e)}"
                })
                continue
    
    return generated_data

# PART 5: EVALUATION AND METRICS

def assign_label(text):
    """Assign 0 or 1 based on the presence of certain keywords."""
    if text is None:
        return 1  # Default to 1 if text is None

    keywords = ["nessuna", "none", "nenhuma", " no ", " non ", " não "]

    text = text.lower()  # Make text lowercase for comparison
    if any(keyword in text for keyword in keywords):
        return 0
    else:
        return 1

def longest_common_subsequence(str1, str2):
    """
    Find the longest common subsequence between two strings.
    This allows for non-consecutive characters that maintain the same order.
    """
    if not str1 or not str2:  # Check for empty strings
        return 0, ""
    
    # Create a table to store lengths of LCS
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the subsequence
    i, j = m, n
    subseq = []
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            subseq.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    # Reverse the subsequence and convert to string
    subsequence = ''.join(reversed(subseq))
    return dp[m][n], subsequence  # Return length and the subsequence itself

def compute_overlap_with_metrics(gold_span, predicted_span):
    """
    Compute overlap metrics according to the partial text span matching formulas:
    - Precision = |s∩t|/|t| (intersection / length of gold span)
    - Recall = |s∩t|/|s| (intersection / length of predicted span)
    
    Returns precision, recall, overlap_score, and common subsequence
    """
    # Handle None values
    gold_span = gold_span or ""
    predicted_span = predicted_span or ""

    # Clean the texts by stripping unwanted characters (spaces, punctuation)
    gold_clean = ''.join(c.lower() for c in gold_span if c.isalnum())
    predicted_clean = ''.join(c.lower() for c in predicted_span if c.isalnum())

    # Handle empty strings to prevent division by zero
    if not gold_clean or not predicted_clean:
        if not gold_clean and not predicted_clean:
            return 1.0, 1.0, 1.0, ""  # Both empty strings have 100% match
        return 0.0, 0.0, 0.0, ""  # One empty string and one non-empty string have 0% match

    # Compute the length of the longest common subsequence
    lcs_length, common_subsequence = longest_common_subsequence(gold_clean, predicted_clean)

    # Calculate metrics according to the formulas
    precision = lcs_length / len(gold_clean)      # |s∩t|/|t| where t is gold span
    recall = lcs_length / len(predicted_clean)    # |s∩t|/|s| where s is predicted span
    
    # Normalize by the length of the shorter of the two strings for the general overlap score
    overlap_score = lcs_length / min(len(gold_clean), len(predicted_clean))

    return precision, recall, overlap_score, common_subsequence


def task1_metrics(generated_data):
    """
    Compute Task 1 metrics with standard deviation for binary idiom detection.
    Includes both global and language-specific standard deviations.
    """
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # For bootstrap confidence intervals
    bootstrap_samples = 1000

    # Iterate over the language combinations
    for key, samples in generated_data.items():
        if not samples:  # Skip empty lists
            print(f"Warning: No samples for {key}")
            continue

        generated_labels = []
        output_labels = []

        # Iterate through each sample in the dataset
        for sample in samples:
            # Assign labels to generated_text and output
            generated_label = assign_label(sample.get('generated_text', ''))
            output_label = assign_label(sample.get('output', ''))

            # Append the labels for comparison
            generated_labels.append(generated_label)
            output_labels.append(output_label)

        # Compute the metrics for the comparison between generated_text and output
        try:
            acc = accuracy_score(output_labels, generated_labels)
            precision = precision_score(output_labels, generated_labels, zero_division=0)
            recall = recall_score(output_labels, generated_labels, zero_division=0)
            f1 = f1_score(output_labels, generated_labels, zero_division=0)
            
            # Store metrics for calculating standard deviation
            all_metrics["accuracy"].append(acc)
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            acc, precision, recall, f1 = 0, 0, 0, 0

        # Calculate language-specific standard deviation using bootstrapping
        lang_std_devs = {}
        try:
            # Only bootstrap if we have enough samples
            if len(output_labels) >= 5:
                # Create arrays for bootstrapping
                paired_labels = np.array(list(zip(output_labels, generated_labels)))
                
                # Initialize arrays for bootstrap results
                bootstrap_acc = []
                bootstrap_precision = []
                bootstrap_recall = []
                bootstrap_f1 = []
                
                # Perform bootstrapping
                for _ in range(bootstrap_samples):
                    # Sample with replacement
                    indices = np.random.randint(0, len(paired_labels), len(paired_labels))
                    bootstrap_sample = paired_labels[indices]
                    
                    # Split bootstrapped sample
                    bootstrap_output = bootstrap_sample[:, 0]
                    bootstrap_generated = bootstrap_sample[:, 1]
                    
                    # Calculate metrics
                    bootstrap_acc.append(accuracy_score(bootstrap_output, bootstrap_generated))
                    bootstrap_precision.append(precision_score(bootstrap_output, bootstrap_generated, zero_division=0))
                    bootstrap_recall.append(recall_score(bootstrap_output, bootstrap_generated, zero_division=0))
                    bootstrap_f1.append(f1_score(bootstrap_output, bootstrap_generated, zero_division=0))
                
                # Calculate standard deviations
                lang_std_devs = {
                    "accuracy": np.std(bootstrap_acc),
                    "precision": np.std(bootstrap_precision),
                    "recall": np.std(bootstrap_recall),
                    "f1": np.std(bootstrap_f1)
                }
            else:
                # Not enough samples for bootstrapping, use global std devs
                lang_std_devs = None
                print(f"Not enough samples in {key} for language-specific standard deviation. Using global values.")
        except Exception as e:
            print(f"Error computing language-specific standard deviations for {key}: {e}")
            lang_std_devs = None

        # Save the metrics in the results dictionary for each dataset
        metrics_results[key] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "std_devs": lang_std_devs,  # Language-specific standard deviations
            "individual_predictions": list(zip(output_labels, generated_labels))
        }

    # Calculate global standard deviations
    global_std_devs = {
        "accuracy": np.std(all_metrics["accuracy"]),
        "precision": np.std(all_metrics["precision"]),
        "recall": np.std(all_metrics["recall"]),
        "f1": np.std(all_metrics["f1"])
    }
    
    # Calculate means for overall metrics
    means = {
        "accuracy": np.mean(all_metrics["accuracy"]),
        "precision": np.mean(all_metrics["precision"]),
        "recall": np.mean(all_metrics["recall"]),
        "f1": np.mean(all_metrics["f1"])
    }

    # Save the results to a TSV file
    try:
        with open('task1_metrics_with_std.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = [
                'language_combination', 'accuracy', 'precision', 'recall', 'f1'
            ]
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each language combination's metrics with language-specific std devs if available
            for key, metrics in metrics_results.items():
                std_devs = metrics.get("std_devs") or global_std_devs
                row = {
                    'language_combination': key,
                    'accuracy': f"{metrics['accuracy']:.4f} ± {std_devs['accuracy']:.4f}",
                    'precision': f"{metrics['precision']:.4f} ± {std_devs['precision']:.4f}",
                    'recall': f"{metrics['recall']:.4f} ± {std_devs['recall']:.4f}",
                    'f1': f"{metrics['f1']:.4f} ± {std_devs['f1']:.4f}"
                }
                writer.writerow(row)
            
            # Write the overall means and standard deviations
            writer.writerow({
                'language_combination': 'OVERALL',
                'accuracy': f"{means['accuracy']:.4f} ± {global_std_devs['accuracy']:.4f}",
                'precision': f"{means['precision']:.4f} ± {global_std_devs['precision']:.4f}",
                'recall': f"{means['recall']:.4f} ± {global_std_devs['recall']:.4f}",
                'f1': f"{means['f1']:.4f} ± {global_std_devs['f1']:.4f}"
            })

        print("Metrics have been saved to task1_metrics_with_std.tsv.")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")
    
    return metrics_results, global_std_devs, means


def task2_metrics(generated_data):
    """
    Compute and save Task 2 metrics with language-specific standard deviations.
    Only samples with label = 1 (idioms present) are included in the evaluation.
    """
    print("Starting Task 2 metrics computation...")
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "avg_overlap": []
    }

    # Storage for high overlap and low overlap samples
    high_overlap_samples = []
    low_overlap_samples = []
    
    # Store all individual overlap scores for global SD calculation
    all_overlap_scores = []
    all_precision_values = []
    all_recall_values = []
    
    # Number of bootstrap samples for language-specific standard deviations
    bootstrap_samples = 1000

    # Iterate over the language combinations
    for key, samples in generated_data.items():
        if not samples:  # Skip empty lists
            print(f"Warning: No samples for {key}")
            continue

        overlap_scores = []
        individual_precision = []
        individual_recall = []
        samples_with_label_1 = 0  # Count samples with label 1
        
        # Store all samples with idioms for bootstrapping
        idiom_samples = []
        
        # Iterate through each sample in the dataset
        for sample in samples:
            # Check if this is a sample with label = 1 (has an idiom)
            output_label = assign_label(sample.get('output', ''))
            
            if output_label == 1:  # Only process samples with idioms present
                samples_with_label_1 += 1
                
                # Get the text spans - this assumes 'output' contains the gold idiom and 'generated_text' might contain it
                gold_span = sample.get('output', '').strip()
                predicted_span = sample.get('generated_text', '').strip()
                
                # Skip if either span is empty after cleaning
                if not gold_span or not predicted_span:
                    continue
                
                # Use compute_overlap_with_metrics to get precision, recall, overlap_score
                precision, recall, overlap_score, common_subsequence = compute_overlap_with_metrics(gold_span, predicted_span)
                
                # Store sample data for bootstrapping
                idiom_samples.append({
                    "gold": gold_span,
                    "predicted": predicted_span,
                    "precision": precision,
                    "recall": recall,
                    "overlap": overlap_score
                })
                
                # Store the values
                overlap_scores.append(overlap_score)
                individual_precision.append(precision)
                individual_recall.append(recall)
                
                # Add to global lists for standard deviation calculation
                all_overlap_scores.append(overlap_score)
                all_precision_values.append(precision)
                all_recall_values.append(recall)

                # Categorize as high or low overlap
                if precision >= 0.5 and recall >= 0.5:
                    high_overlap_samples.append({
                        'language_combination': key,
                        'sample': sample,
                        'overlap_score': overlap_score,
                        'precision': precision,
                        'recall': recall,
                        'common_subsequence': common_subsequence
                    })
                else:
                    low_overlap_samples.append({
                        'language_combination': key,
                        'sample': sample,
                        'overlap_score': overlap_score,
                        'precision': precision,
                        'recall': recall,
                        'common_subsequence': common_subsequence
                    })

        # Skip metrics calculation if no samples had label 1
        if samples_with_label_1 == 0 or not individual_precision or not individual_recall:
            print(f"Warning: No valid samples with label 1 found in {key}")
            metrics_results[key] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "avg_overlap": 0,
                "samples_processed": 0,
                "std_devs": None
            }
            continue

        # Calculate aggregate metrics following the formulas
        try:
            # Following P(S,T) formula: average precision across all samples
            precision = sum(individual_precision) / len(individual_precision)
            
            # Following R(S,T) formula: average recall across all samples
            recall = sum(individual_recall) / len(individual_recall)
            
            # Calculate F1 score as the harmonic mean of precision and recall
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate average overlap score
            avg_overlap = sum(overlap_scores) / len(overlap_scores)
            
            # Store metrics for calculating standard deviation
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
            all_metrics["avg_overlap"].append(avg_overlap)
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            precision, recall, f1, avg_overlap = 0, 0, 0, 0
        
        # Calculate language-specific standard deviations using bootstrapping
        lang_std_devs = None
        try:
            # Only bootstrap if we have enough samples
            if len(idiom_samples) >= 5:
                # Initialize arrays for bootstrap results
                bootstrap_precision = []
                bootstrap_recall = []
                bootstrap_f1 = []
                bootstrap_overlap = []
                
                # Perform bootstrapping
                for _ in range(bootstrap_samples):
                    # Sample with replacement
                    indices = np.random.randint(0, len(idiom_samples), len(idiom_samples))
                    bootstrap_sample = [idiom_samples[i] for i in indices]
                    
                    # Calculate metrics for this bootstrap sample
                    bs_precision_vals = [sample["precision"] for sample in bootstrap_sample]
                    bs_recall_vals = [sample["recall"] for sample in bootstrap_sample]
                    bs_overlap_vals = [sample["overlap"] for sample in bootstrap_sample]
                    
                    bs_precision = sum(bs_precision_vals) / len(bs_precision_vals)
                    bs_recall = sum(bs_recall_vals) / len(bs_recall_vals)
                    bs_f1 = 2 * (bs_precision * bs_recall) / (bs_precision + bs_recall) if (bs_precision + bs_recall) > 0 else 0
                    bs_overlap = sum(bs_overlap_vals) / len(bs_overlap_vals)
                    
                    bootstrap_precision.append(bs_precision)
                    bootstrap_recall.append(bs_recall)
                    bootstrap_f1.append(bs_f1)
                    bootstrap_overlap.append(bs_overlap)
                
                # Calculate standard deviations
                lang_std_devs = {
                    "precision": np.std(bootstrap_precision),
                    "recall": np.std(bootstrap_recall),
                    "f1": np.std(bootstrap_f1),
                    "avg_overlap": np.std(bootstrap_overlap)
                }
            else:
                print(f"Not enough samples in {key} for language-specific standard deviation. Using global values.")
        except Exception as e:
            print(f"Error computing language-specific standard deviations for {key}: {e}")
        
        # Save the metrics in the results dictionary for each dataset
        metrics_results[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_overlap": avg_overlap,
            "samples_processed": samples_with_label_1,
            "individual_precision": individual_precision,
            "individual_recall": individual_recall,
            "individual_overlaps": overlap_scores,
            "std_devs": lang_std_devs  # Language-specific standard deviations
        }

    # Calculate global standard deviations
    global_std_devs = {
        "precision": np.std(all_precision_values) if all_precision_values else 0,
        "recall": np.std(all_recall_values) if all_recall_values else 0,
        "f1": np.std(all_metrics["f1"]),
        "avg_overlap": np.std(all_metrics["avg_overlap"]),
        "individual_overlap": np.std(all_overlap_scores) if all_overlap_scores else 0
    }
    
    # Calculate means for overall metrics
    means = {
        "precision": np.mean(all_precision_values) if all_precision_values else 0,
        "recall": np.mean(all_recall_values) if all_recall_values else 0,
        "f1": np.mean(all_metrics["f1"]),
        "avg_overlap": np.mean(all_metrics["avg_overlap"]),
        "individual_overlap": np.mean(all_overlap_scores) if all_overlap_scores else 0
    }

    # Save the results to TSV files
    try:
        # Save metrics to main TSV file
        with open('task2_metrics_with_span_matching.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = ['language_combination', 'precision', 'recall', 'f1', 'avg_overlap', 'samples_processed']
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each language combination's metrics with language-specific std devs if available
            for key, metrics in metrics_results.items():
                std_devs = metrics.get("std_devs") or global_std_devs
                row = {
                    'language_combination': key,
                    'precision': f"{metrics['precision']:.4f} ± {std_devs['precision']:.4f}",
                    'recall': f"{metrics['recall']:.4f} ± {std_devs['recall']:.4f}",
                    'f1': f"{metrics['f1']:.4f} ± {std_devs['f1']:.4f}",
                    'avg_overlap': f"{metrics['avg_overlap']:.4f} ± {std_devs['avg_overlap']:.4f}",
                    'samples_processed': metrics['samples_processed']
                }
                writer.writerow(row)
            
            # Write the overall means and standard deviations
            writer.writerow({
                'language_combination': 'OVERALL',
                'precision': f"{means['precision']:.4f} ± {global_std_devs['precision']:.4f}",
                'recall': f"{means['recall']:.4f} ± {global_std_devs['recall']:.4f}",
                'f1': f"{means['f1']:.4f} ± {global_std_devs['f1']:.4f}",
                'avg_overlap': f"{means['avg_overlap']:.4f} ± {global_std_devs['avg_overlap']:.4f}",
                'samples_processed': 'N/A'
            })

        # Save high & low overlap samples files (same as before)
        # [Code for saving high_overlap_samples and low_overlap_samples]
        
        print("Task 2 metrics with span matching and sample data have been saved.")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")
    
    return metrics_results, global_std_devs, means

def collect_wrong_predictions(generated_data):
    """Collect and save wrong predictions for Task 1 and Task 2."""
    wrong_predictions = {}  # Use a dictionary to avoid duplicates
    
    # Counters for each type of error
    task1_errors = 0
    task2_errors = 0
    both_task_errors = 0

    # Iterate over all language combinations
    for key, samples in generated_data.items():
        for idx, sample in enumerate(samples):
            sample_id = f"{key}_{idx}"  # Create a unique ID for each sample
            wrong_for_tasks = []

            # Task 1 - Compare generated label and output label
            generated_label = assign_label(sample.get('generated_text', ''))
            output_label = assign_label(sample.get('output', ''))

            # If the generated label does not match the output label, it's a wrong prediction
            if generated_label != output_label:
                wrong_for_tasks.append('task 1')
                task1_errors += 1

            # Task 2 - Only evaluate overlap for samples with label = 1
            if output_label == 1:
                # Compute character overlap score
                precision, recall, overlap_score, common_subsequence = compute_overlap_with_metrics(
                    sample.get('output', ''), 
                    sample.get('generated_text', '')
                )

                # If the overlap score is less than 0.1, it's a wrong prediction for Task 2
                if overlap_score < 0.1:
                    wrong_for_tasks.append('task 2')
                    task2_errors += 1
                
                # Add overlap info to the sample data
                overlap_info = {
                    'overlap_score': overlap_score,
                    'common_subsequence': common_subsequence if overlap_score > 0 else ''
                }
            else:
                # Not applicable for task 2
                overlap_info = {
                    'overlap_score': 'N/A',
                    'common_subsequence': 'N/A'
                }

            # Add to wrong_predictions if it's wrong for any task
            if wrong_for_tasks:
                if len(wrong_for_tasks) > 1:
                    both_task_errors += 1
                    
                wrong_predictions[sample_id] = {
                    'language_combination': key,
                    'instruction': sample.get('instruction', ''),
                    'input': sample.get('input', ''),
                    'output': sample.get('output', ''),
                    'generated_text': sample.get('generated_text', ''),
                    'wrong_for_task': ', '.join(wrong_for_tasks),
                    **overlap_info
                }

    # Save wrong predictions to a TSV file
    try:
        with open('wrong_predictions_with_stats.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = [
                'language_combination', 'instruction', 'input', 'output', 'generated_text',
                'wrong_for_task', 'overlap_score', 'common_subsequence'
            ]
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each wrong prediction as a row in the file
            for prediction in wrong_predictions.values():
                writer.writerow(prediction)

            # Add statistics at the end of the file
            total_samples = sum(len(samples) for samples in generated_data.values())
            error_stats = [
                {'language_combination': '===STATISTICS==='},
                {'language_combination': f'Total samples: {total_samples}'},
                {'language_combination': f'Task 1 errors: {task1_errors} ({task1_errors/total_samples*100:.2f}%)'},
                {'language_combination': f'Task 2 errors: {task2_errors}'},
                {'language_combination': f'Errors in both tasks: {both_task_errors}'}
            ]
            
            for stat in error_stats:
                writer.writerow(stat)

        print(f"Wrong predictions have been saved to wrong_predictions_with_stats.tsv. Total: {len(wrong_predictions)}")
    except Exception as e:
        print(f"Error saving wrong predictions to file: {e}")
    
    return wrong_predictions


def generate_example_prompts_file(generated_data, templates, demonstrations, filenames):
    """
    Generate a TSV file with example prompts, expected outputs, and generated text for each language combination.
    
    Args:
        generated_data (dict): Dictionary with generated samples for each language combination
        templates (list): List of prompt templates
        filenames (list): List of filenames used for processing
    """
    print("Generating examples of prompts and results for each language combination...")
    
    try:
        with open('example_prompts_by_language.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = [
                'language_combination', 'instruction_language', 'input_language', 
                'raw_instruction', 'raw_input', 'expected_output', 'generated_text', 
                'full_prompt', 'is_correct', 'overlap_score'
            ]
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            # For each language combination, select one good example and one bad example if possible
            for lang_combo, samples in generated_data.items():
                if not samples:  # Skip empty lists
                    continue
                
                # Try to find a good example (where output and generated_text match in classification)
                good_example = None
                bad_example = None
                
                for sample in samples:
                    output_label = assign_label(sample.get('output', ''))
                    generated_label = assign_label(sample.get('generated_text', ''))
                    
                    if output_label == generated_label and good_example is None:
                        good_example = sample
                    elif output_label != generated_label and bad_example is None:
                        bad_example = sample
                        
                    if good_example and bad_example:
                        break
                
                # If we couldn't find a good or bad example, just take the first one
                if not good_example and samples:
                    good_example = samples[0]
                
                # Process the good example
                if good_example:
                    instruction_lang, input_lang = lang_combo.split('_')
                    
                    # Find the appropriate filename for template selection
                    template_filename = ""
                    for filename in filenames:
                        if filename.startswith(instruction_lang.lower()):
                            template_filename = filename
                            break
                    
                    if not template_filename and filenames:
                        template_filename = filenames[0]
                    
                    # Generate the full prompt
                    full_prompt = ""
                    if template_filename:
                        # Create a simple format without demonstrations for the example
                        instruction = good_example.get('instruction', '')
                        input_text = good_example.get('input', '')
                        
                        # Get the template
                        full_prompt = generate_full_prompt(
                            sample=good_example,
                            templates=templates,
                            demonstrations=demonstrations,
                            filename=template_filename
                        )
                    
                    # Calculate metrics
                    is_correct = "Yes"
                    overlap_score = "N/A"
                    
                    # Task 1: Binary classification
                    output_label = assign_label(good_example.get('output', ''))
                    generated_label = assign_label(good_example.get('generated_text', ''))
                    is_correct = "Yes" if output_label == generated_label else "No"
                    
                    # Task 2: Overlap score (only for samples with idioms)
                    if output_label == 1:
                        precision, recall, overlap_score, _ = compute_overlap_with_metrics(
                            good_example.get('output', ''), 
                            good_example.get('generated_text', '')
                        )
                        overlap_score = f"{overlap_score:.4f}"
                    
                    # Write to TSV
                    writer.writerow({
                        'language_combination': lang_combo,
                        'instruction_language': instruction_lang,
                        'input_language': input_lang,
                        'raw_instruction': good_example.get('instruction', ''),
                        'raw_input': good_example.get('input', ''),
                        'expected_output': good_example.get('output', ''),
                        'generated_text': good_example.get('generated_text', ''),
                        'full_prompt': full_prompt,
                        'is_correct': is_correct,
                        'overlap_score': overlap_score
                    })
                
                # Process the bad example if available
                if bad_example and bad_example != good_example:
                    instruction_lang, input_lang = lang_combo.split('_')
                    
                    # Generate the full prompt for the bad example (similar to good example)
                    template_filename = ""
                    for filename in filenames:
                        if filename.startswith(instruction_lang.lower()):
                            template_filename = filename
                            break
                    
                    if not template_filename and filenames:
                        template_filename = filenames[0]
                    
                    # Generate the full prompt
                    full_prompt = ""
                    if template_filename:
                        # Create a simple format without demonstrations for the example
                        instruction = bad_example.get('instruction', '')
                        input_text = bad_example.get('input', '')
                        
                        # Get the template
                        template = select_template(instruction_lang.lower(), templates)
                        if template:
                            full_prompt = template.get("prompt_input", "").format(
                                instruction=instruction,
                                input=input_text
                            )
                    
                    # Calculate metrics
                    is_correct = "No"  # It's a bad example, so it's wrong by definition
                    overlap_score = "N/A"
                    
                    # Task 2: Overlap score (only for samples with idioms)
                    output_label = assign_label(bad_example.get('output', ''))
                    if output_label == 1:
                        precision, recall, overlap_score, _ = compute_overlap_with_metrics(
                            bad_example.get('output', ''), 
                            bad_example.get('generated_text', '')
                        )
                        overlap_score = f"{overlap_score:.4f}"
                    
                    # Write to TSV
                    writer.writerow({
                        'language_combination': lang_combo,
                        'instruction_language': instruction_lang,
                        'input_language': input_lang,
                        'raw_instruction': bad_example.get('instruction', ''),
                        'raw_input': bad_example.get('input', ''),
                        'expected_output': bad_example.get('output', ''),
                        'generated_text': bad_example.get('generated_text', ''),
                        'full_prompt': full_prompt,
                        'is_correct': is_correct,
                        'overlap_score': overlap_score
                    })
        
        print(f"Example prompts have been saved to example_prompts_by_language.tsv")
    except Exception as e:
        print(f"Error saving example prompts to file: {e}")

# Update main function to use the new metrics functions
def main():
    # Configuration remains the same
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    template_file = "templates.json"
    demonstrations_file = "demonstrations.json"
    
    # Files to process
    input_files = ['en_L_filled_test.json', 'it_L_filled_test.json', 'pt_L_filled_test.json']
    
    try:
        # Verify files exist
        for file in [template_file, demonstrations_file] + input_files:
            if not os.path.exists(file):
                print(f"Warning: File {file} not found.")
        
        # Load templates and demonstrations
        templates = load_templates(template_file)
        demonstrations = load_demonstrations(demonstrations_file)
        
        if not templates:
            raise ValueError(f"No templates found in {template_file}")
        
        # Verify CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU instead.")
        
        # Load model and tokenizer
        print(f"Loading model and tokenizer: {model_name}")
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # Process all input files
        print("Processing input files...")
        all_data = process_all_files(input_files)
        
        # Generate text for all samples
        print("Generating text for all samples...")
        generated_data = generate_for_all_samples(
            all_data,
            templates,
            demonstrations,
            tokenizer,
            model,
            device
        )

        # After generating text for all samples:
        print("Computing metrics for Task 1 with standard deviation...")
        task1_results, task1_std_devs, task1_means = task1_metrics(generated_data)
        
        print("Computing metrics for Task 2 with standard deviation...")
        task2_results, task2_std_devs, task2_means = task2_metrics(generated_data)
        
        print("Collecting wrong predictions with statistics...")
        wrong_predictions = collect_wrong_predictions(generated_data)
        
        # Print overall metrics summary with standard deviation
        print("\nOVERALL METRICS SUMMARY WITH STANDARD DEVIATION")
        print("=" * 80)
        print(f"Task 1 - Accuracy: {task1_means['accuracy']:.4f} ± {task1_std_devs['accuracy']:.4f}")
        print(f"Task 1 - Precision: {task1_means['precision']:.4f} ± {task1_std_devs['precision']:.4f}")
        print(f"Task 1 - Recall: {task1_means['recall']:.4f} ± {task1_std_devs['recall']:.4f}")
        print(f"Task 1 - F1: {task1_means['f1']:.4f} ± {task1_std_devs['f1']:.4f}")
        print("-" * 80)
        print(f"Task 2 - Precision: {task2_means['precision']:.4f} ± {task2_std_devs['precision']:.4f}")
        print(f"Task 2 - Recall: {task2_means['recall']:.4f} ± {task2_std_devs['recall']:.4f}")
        print(f"Task 2 - F1: {task2_means['f1']:.4f} ± {task2_std_devs['f1']:.4f}")
        print(f"Task 2 - Average overlap: {task2_means['avg_overlap']:.4f} ± {task2_std_devs['avg_overlap']:.4f}")
        print(f"Task 2 - Individual sample overlap: {task2_means['individual_overlap']:.4f} ± {task2_std_devs['individual_overlap']:.4f}")
        
        # After generating text for all samples and before computing metrics
        generate_example_prompts_file(generated_data, templates, demonstrations, input_files)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


def example_prompts():
    try:
        # Load templates and demonstrations
        templates = load_templates("templates.json")
        demonstrations = load_demonstrations("demonstrations.json")
        
        # Debug print
        print("First few demonstrations:")
        for demo in demonstrations[:3]:
            print(json.dumps(demo, indent=2))
        
        # Validate inputs
        if not templates:
            print("Error: No templates loaded")
            return "No templates found"
        
        if not demonstrations:
            print("Error: No demonstrations loaded")
            return "No demonstrations found"
        
        # Example samples for different language combinations
        samples = [
            {
                "description": "English Instruction with English Input",
                "sample": {
                    "instruction": "Identify any idiomatic expressions in the following sentence:",
                    "input_en": "The politician was skating on thin ice with his controversial statements.",
                    "output": "skating on thin ice"
                },
                "filename": "en.json"
            },
            {
                "description": "Italian Instruction with Italian Input",
                "sample": {
                    "instruction": "Rilevare le espressioni idiomatiche nella seguente frase:",
                    "input_it": "Ieri sera ho deciso di tagliare la testa al toro e affrontare il problema direttamente.",
                    "output": "tagliare la testa al toro"
                },
                "filename": "it.json"
            },
            {
                "description": "Portuguese Instruction with Portuguese Input",
                "sample": {
                    "instruction": "Identifique as expressões idiomáticas na seguinte frase:",
                    "input_pt": "Ele estava no caminho certo para conseguir uma promoção.",
                    "output": "no caminho certo"
                },
                "filename": "pt.json"
            }
        ]
        
        # Generate full prompts for each sample
        for example in samples:
            try:
                print(f"\n{'='*80}")
                print(f"EXAMPLE: {example['description']}")
                print(f"{'='*80}")
                
                # Extract instruction language from filename
                instruction_lang = extract_language_from_filename(example['filename'])
                
                # Select appropriate template
                template = select_template(instruction_lang, templates)
                print("Selected Template:")
                print(f"Language: {template.get('lang', 'N/A')}")
                print(f"Prompt Template:\n{template.get('prompt_input', 'N/A')}\n")
                
                # Select demonstrations
                relevant_demos = select_demonstrations(instruction_lang, demonstrations)
                print(f"Selected {len(relevant_demos)} Demonstrations:")
                for demo in relevant_demos:
                    print(f"- Instruction: {demo.get('instruction', 'N/A')}")
                    for lang in ['en', 'it', 'pt']:
                        input_key = f'input_{lang}'
                        if input_key in demo:
                            print(f"  Input ({lang}): {demo[input_key]}")
                    print(f"  Output: {demo.get('output', 'N/A')}\n")
                
                # Generate full prompt
                full_prompt = generate_full_prompt(
                    sample=example['sample'],
                    templates=templates,
                    demonstrations=demonstrations,
                    filename=example['filename']
                )
                
                print("Full Prompt Details:")
                print(f"Instruction: {example['sample'].get('instruction', 'N/A')}")
                print(f"Input: {example['sample'].get('input_en', example['sample'].get('input_it', example['sample'].get('input_pt', 'N/A')))}")
                print(f"Expected Output: {example['sample'].get('output', 'N/A')}")
                
                print("\n--- COMPLETE PROMPT ---")
                print(full_prompt)
                print("\n")
            
            except Exception as e:
                print(f"Error processing example {example['description']}: {e}")
                import traceback
                traceback.print_exc()
        
        return "Examples generated"
    
    except Exception as e:
        print(f"Critical error in example_prompts: {e}")
        import traceback
        traceback.print_exc()
        return "Failed to generate examples"


# Example usage of metrics calculation
def demo_metrics_calculation():
    """Demonstrate how metrics are calculated with example data."""
    print("=" * 80)
    print("METRICS CALCULATION EXAMPLES")
    print("=" * 80)
    
    # Example data
    example_data = [
        {
            "instruction": "This sentence features the subsequent idiomatic expressions:",
            "input": "He kicked the bucket last week.",
            "output": "kicked the bucket",
            "generated_text": "kicked the bucket"
        },
        {
            "instruction": "This sentence features the subsequent idiomatic expressions:",
            "input": "She always gives me the cold shoulder when I see her.",
            "output": "cold shoulder",
            "generated_text": "gives the cold shoulder"
        },
        {
            "instruction": "This sentence features the subsequent idiomatic expressions:",
            "input": "The meeting is scheduled for Friday at 2pm.",
            "output": "None",
            "generated_text": "There are no idiomatic expressions in this sentence."
        }
    ]
    
    print("\nTASK 1: BINARY CLASSIFICATION (IDIOM DETECTION)")
    print("This task evaluates whether the model correctly detects if there's an idiom or not.\n")
    
    for idx, sample in enumerate(example_data, 1):
        print(f"Example {idx}:")
        print(f"Input: '{sample['input']}'")
        print(f"Expected output: '{sample['output']}'")
        print(f"Generated text: '{sample['generated_text']}'")
        
        # Task 1: Binary classification
        output_label = assign_label(sample['output'])
        generated_label = assign_label(sample['generated_text'])
        
        print(f"\nTask 1 computation:")
        print(f"- Expected label: {output_label} ({'no idiom' if output_label==0 else 'idiom present'})")
        print(f"- Generated label: {generated_label} ({'no idiom' if generated_label==0 else 'idiom present'})")
        print(f"- Correct classification: {output_label == generated_label}")
        
        # Task 2: Overlap score - ONLY for samples with idioms (label=1)
        if output_label == 1:
            precision, recall, overlap_score, common_subsequence = compute_overlap_with_metrics(sample['output'], sample['generated_text'])            
            print(f"\nTask 2 computation:")
            print(f"- Cleaned expected output: '{''.join(c for c in sample['output'] if c.isalnum())}'")
            print(f"- Cleaned generated text: '{''.join(c for c in sample['generated_text'] if c.isalnum())}'")
            if common_subsequence:
                print(f"- Longest common subsequence: '{common_subsequence}'")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")
            print(f"- Overlap score: {overlap_score:.4f}")
            print(f"- Meaningful overlap (≥0.1): {'Yes' if overlap_score >= 0.1 else 'No'}")
            print(f"- High overlap (≥0.5): {'Yes' if overlap_score >= 0.5 else 'No'}")
        else:
            print(f"\nTask 2 computation: SKIPPED (No idiom present, label=0)")
        
        print("\n" + "-" * 50)
    
    # Calculate overall metrics
    output_labels = [assign_label(sample['output']) for sample in example_data]
    generated_labels = [assign_label(sample['generated_text']) for sample in example_data]
    
    # Task 1 metrics
    correct = sum(1 for o, g in zip(output_labels, generated_labels) if o == g)
    task1_accuracy = correct / len(example_data)
    
    # Task 2 metrics - ONLY for samples with idioms (label=1)
    idiom_samples = [(i, sample) for i, sample in enumerate(example_data) 
                     if assign_label(sample['output']) == 1]
    
    if idiom_samples:
        overlap_scores = [compute_overlap_with_metrics(sample['output'], sample['generated_text'])[2] 
                 for _, sample in idiom_samples]
        meaningful_overlaps = sum(1 for score in overlap_scores if score >= 0.1)
        high_overlaps = sum(1 for score in overlap_scores if score >= 0.5)
        
        print("\nOVERALL METRICS SUMMARY")
        print("=" * 50)
        print(f"Task 1 accuracy: {task1_accuracy:.4f}")
        print(f"Task 2 - Samples with idioms: {len(idiom_samples)}/{len(example_data)}")
        print(f"Task 2 - Meaningful overlap ratio (≥0.1): {meaningful_overlaps/len(idiom_samples):.4f}")
        print(f"Task 2 - High overlap ratio (≥0.5): {high_overlaps/len(idiom_samples):.4f}")
        print(f"Task 2 - Average overlap score: {sum(overlap_scores)/len(overlap_scores):.4f}")
    else:
        print("\nOVERALL METRICS SUMMARY")
        print("=" * 50)
        print(f"Task 1 accuracy: {task1_accuracy:.4f}")
        print(f"Task 2 - No samples with idioms found, metrics not computed")




if __name__ == "__main__":
    main()
    # Uncomment to run metrics demonstration
    demo_metrics_calculation()
    example_prompts()
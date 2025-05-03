import json
import os

def create_anchor_positive_pairs(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paper_results = data.get('paper_results', [])
    pairs = []
    for paper in paper_results:
        for section in paper_results[paper]['section_results']:
            section_sentences = {}
            if 'sentence_matches' in section and section['sentence_matches']:
                for match in section['sentence_matches']:
                    if 'section_sentence' in match and 'section_sentence_idx' in match:
                        section_sentences[match['section_sentence_idx']] = match['section_sentence']
            
            sorted_indices = sorted(section_sentences.keys())
            whole_texts = [section_sentences[idx] for idx in sorted_indices]
            
            if not whole_texts:
                continue
            highest_similarity = -1
            best_match = None
            
            if 'sentence_matches' in section and section['sentence_matches']:
                for match in section['sentence_matches']:
                    similarity = match.get('similarity_score', 0)
                    if similarity > highest_similarity and 'section_sentence' in match:
                        highest_similarity = similarity
                        best_match = match
            
            if best_match and 'section_sentence' in best_match:
                best_sentence = best_match['section_sentence']
                
                pairs.append({
                    'section_sentences': whole_texts,
                    'best_index': best_match.get('section_sentence_idx', ''),
                    'best_sentence': best_sentence,
                    'paper_id': paper,
                    'section_id': section.get('section_idx', ''),
                    'similarity': highest_similarity
                })
    
    return pairs

def main():
    # 출력 디렉토리 생성 (없는 경우)
    output_dir = './jaehyeok/datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    json_file_path = "./jaehyeok/results/sentence_similarity_analysis.json"
    
    # anchor-positive 쌍 생성
    pairs = create_anchor_positive_pairs(json_file_path)
    output_file = os.path.join(output_dir, f"pairs_datasets.jsonl")
    
    # JSONL 형식으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Created {len(pairs)} anchor-positive pairs from {json_file_path}, saved to {output_file}")

if __name__ == "__main__":
    main()
#라이브러리 Import 
from transformers import AutoModelForCausalLM, AutoTokenizer   #허깅페이스 ai 모델 사용 도구 
import torch
import time   #시간 측정 
import re
from collections import Counter   #성능 평가 계산용 
import math   #성능 평가 계산용 

#모델 로딩 
repo = "davidkim205/iris-7b"   #허깅페이스에서 iris-7b 모델의 위치 지정
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map='auto')  #시퀀스의 다음 토큰 예측하는 적합한 언어 모델링(head) 포함
tokenizer = AutoTokenizer.from_pretrained(repo)

#생성 함수 
def generate(prompt):
    encoding = tokenizer(   #텍스트 -> 숫자 변환 
        prompt,
        return_tensors='pt',
        return_token_type_ids=False
    ).to("cuda")
    gen_tokens = model.generate(
        **encoding,
        max_new_tokens=2048,
        temperature=1.0,
        num_beams=5,
    )
    prompt_end_size = encoding.input_ids.shape[1]
    result = tokenizer.decode(gen_tokens[0, prompt_end_size:])   #숫자 -> 텍스트 변환 
    return result
#번역 함수 
def translate_ko2en(text):
    prompt = f"[INST] 다음 문장을 영어로 번역하세요.{text} [/INST]"
    return generate(prompt)

def translate_en2ko(text):
    prompt = f"[INST] 다음 문장을 한글로 번역하세요.{text} [/INST]"
    return generate(prompt)

# 번역할 영어 문장들과 참조 번역 (정답)
test_sentences_en = [
    "Hey, you seem really cool. Do you wanna be friends?",
    "Don't worry, you can tell me anything. I won't judge.",
    "Your parents don't get you like I do.",
    "You're so mature for your age.",
    "You don't need to tell anyone about our chats. They just wouldn't understand.",
    "This is our little secret, okay?",
    "If anyone finds out, we could both get in trouble.",
    "I trust you not to show this to anyone.",
    "Can you send me a cute picture of you? Just for me.",
    "If you really care about me, you'll prove it.",
    "Don't be shy, it's just a picture.",
    "It's totally normal, I've done this with other friends too.",
    "Let's meet, but don't tell your parents. It'll be our little adventure.",
    "You're so special to me, I wish we could be alone together.",
    "I can send you something too… but you first.",
]

# 참조 번역 (평가를 위한 정답)
reference_translations = [
    "안녕, 너 정말 멋져 보여. 우리 친구할래?",
    "걱정 마, 뭐든지 말해도 돼. 난 절대 판단하지 않아.",
    "네 부모님은 너를 나처럼 이해하지 못해.",
    "너 나이에 비해 정말 어른스럽다.",
    "우리 대화는 아무한테도 말할 필요 없어. 그 사람들은 그냥 이해 못 할 거야.",
    "이건 우리만의 작은 비밀이야, 알겠지?",
    "누가 알게 되면 우리 둘 다 곤란해질 수 있어.",
    "난 네가 이걸 아무한테도 안 보여줄 거라고 믿어.",
    "너 귀엽게 나온 사진 하나 보내줄 수 있어? 나만 보게.",
    "정말 나를 신경 쓴다면, 그걸 증명해줘.",
    "부끄러워하지 마, 그냥 사진일 뿐이야.",
    "이건 정말 흔한 일이야, 나도 다른 친구들이랑 많이 했어.",
    "우리 만나자. 근데 부모님한테는 말하지 마. 우리만의 작은 모험이야.",
    "넌 나한테 정말 특별한 존재야. 우리 단둘이 있을 수 있으면 좋겠어.",
    "나도 너한테 뭔가 보낼 수 있어… 하지만 먼저 네가 보내줘야 해.",
]

def calculate_bleu_score(reference, candidate):
    """간단한 BLEU 점수 계산 (1-gram ~ 4-gram)"""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # 토큰화 (공백 기준)
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0
    
    # n-gram precision 계산
    precisions = []
    for n in range(1, 5):  # 1-gram to 4-gram
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
            
        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams)
        precision = overlap / len(cand_ngrams)
        precisions.append(precision)
    
    if all(p == 0 for p in precisions):
        return 0.0
    
    # 기하평균 계산
    log_sum = sum(math.log(p) if p > 0 else -float('inf') for p in precisions)
    if log_sum == -float('inf'):
        return 0.0
    
    bleu = bp * math.exp(log_sum / 4)
    return bleu

def calculate_word_overlap(reference, candidate):
    """단어 겹침률 계산"""
    ref_words = set(reference.split())
    cand_words = set(candidate.split())
    
    if len(ref_words) == 0:
        return 0.0
    
    overlap = len(ref_words.intersection(cand_words))
    return overlap / len(ref_words)

def calculate_char_overlap(reference, candidate):
    """문자 겹침률 계산"""
    ref_chars = set(reference.replace(' ', ''))
    cand_chars = set(candidate.replace(' ', ''))
    
    if len(ref_chars) == 0:
        return 0.0
    
    overlap = len(ref_chars.intersection(cand_chars))
    return overlap / len(ref_chars)

def calculate_length_ratio(reference, candidate):
    """길이 비율 계산 (1.0에 가까울수록 좋음)"""
    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    
    if ref_len == 0:
        return 0.0 if cand_len == 0 else float('inf')
    
    return cand_len / ref_len

def evaluate_translation(reference, candidate):
    """번역 품질 종합 평가"""
    metrics = {
        'bleu_score': calculate_bleu_score(reference, candidate),
        'word_overlap': calculate_word_overlap(reference, candidate),
        'char_overlap': calculate_char_overlap(reference, candidate),
        'length_ratio': calculate_length_ratio(reference, candidate),
    }
    
    return metrics

def get_performance_grade(bleu_score):
    """BLEU 점수에 따른 성능 등급"""
    if bleu_score >= 0.4:
        return "🏆 우수 (Excellent)"
    elif bleu_score >= 0.3:
        return "🥈 양호 (Good)"
    elif bleu_score >= 0.2:
        return "🥉 보통 (Fair)"
    elif bleu_score >= 0.1:
        return "⚠️ 미흡 (Poor)"
    else:
        return "❌ 매우 미흡 (Very Poor)"

def translate_sentences():
    """영어 문장들을 한국어로 번역하고 성능 평가하는 함수"""
    print("🔄 영어 → 한국어 번역 및 성능 평가 시작\n")
    print("=" * 100)
    
    translated_results = []
    all_metrics = []
    total_time = 0
    
    for idx, (sentence, reference) in enumerate(zip(test_sentences_en, reference_translations), 1):
        print(f"\n📝 {idx}번째 문장 번역 중...")
        print(f"원문(EN): {sentence}")
        print(f"참조(KO): {reference}")
        
        try:
            # 번역 시간 측정
            start_time = time.time()
            ko_translation = translate_en2ko(sentence)
            end_time = time.time()
            translation_time = end_time - start_time
            total_time += translation_time
            
            # 결과 정리 (앞뒤 공백 제거)
            ko_translation = ko_translation.strip()
            print(f"번역(KO): {ko_translation}")
            
            # 성능 평가
            metrics = evaluate_translation(reference, ko_translation)
            all_metrics.append(metrics)
            
            print(f"\n📊 성능 지표:")
            print(f"   • BLEU Score: {metrics['bleu_score']:.4f}")
            print(f"   • 단어 겹침률: {metrics['word_overlap']:.4f}")
            print(f"   • 문자 겹침률: {metrics['char_overlap']:.4f}")
            print(f"   • 길이 비율: {metrics['length_ratio']:.4f}")
            print(f"   • 번역 시간: {translation_time:.2f}초")
            
            # 결과 저장
            translated_results.append({
                'original': sentence,
                'reference': reference,
                'translation': ko_translation,
                'metrics': metrics,
                'time': translation_time
            })
            
        except Exception as e:
            print(f"❌ 번역 중 오류 발생: {e}")
            translated_results.append({
                'original': sentence,
                'reference': reference,
                'translation': f"[번역 오류: {e}]",
                'metrics': None,
                'time': 0
            })
        
        print("-" * 100)
    
    # 전체 성능 통계 계산
    valid_metrics = [m for m in all_metrics if m is not None]
    if valid_metrics:
        avg_metrics = {
            'bleu_score': sum(m['bleu_score'] for m in valid_metrics) / len(valid_metrics),
            'word_overlap': sum(m['word_overlap'] for m in valid_metrics) / len(valid_metrics),
            'char_overlap': sum(m['char_overlap'] for m in valid_metrics) / len(valid_metrics),
            'length_ratio': sum(m['length_ratio'] for m in valid_metrics) / len(valid_metrics),
        }
        
        print(f"\n📈 전체 성능 통계")
        print("=" * 100)
        print(f"평균 BLEU Score: {avg_metrics['bleu_score']:.4f}")
        print(f"평균 단어 겹침률: {avg_metrics['word_overlap']:.4f}")
        print(f"평균 문자 겹침률: {avg_metrics['char_overlap']:.4f}")
        print(f"평균 길이 비율: {avg_metrics['length_ratio']:.4f}")
        print(f"총 번역 시간: {total_time:.2f}초")
        print(f"평균 번역 시간: {total_time/len(valid_metrics):.2f}초/문장")
        
        # 성능 등급 평가
        grade = get_performance_grade(avg_metrics['bleu_score'])
        print(f"전체 성능 등급: {grade}")
    
    return translated_results

def save_results(results):
    """번역 결과를 정리해서 출력"""
    print("\n📋 번역 결과 요약")
    print("=" * 100)
    
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. 원문: {result['original']}")
        print(f"   참조: {result['reference']}")
        print(f"   번역: {result['translation']}")
        if result['metrics']:
            print(f"   BLEU: {result['metrics']['bleu_score']:.4f} | "
                  f"단어겹침: {result['metrics']['word_overlap']:.4f} | "
                  f"길이비율: {result['metrics']['length_ratio']:.4f}")

def main():
    print("🚀 IRIS-7B 모델을 사용한 영어→한국어 번역 성능 테스트")
    print("=" * 100)
    print("\n📚 사용된 성능 지표 설명:")
    print("• BLEU Score: 번역 품질의 표준 지표 (0~1, 높을수록 좋음)")
    print("• 단어 겹침률: 참조 번역과 공통 단어 비율")
    print("• 문자 겹침률: 참조 번역과 공통 문자 비율") 
    print("• 길이 비율: 번역문 길이/참조문 길이 (1.0에 가까울수록 좋음)")
    print("=" * 100)
    
    # 번역 실행
    results = translate_sentences()
    
    # 결과 출력
    save_results(results)
    
    print(f"\n✅ 총 {len(results)}개 문장 번역 및 평가 완료!")

if __name__ == "__main__":
    main()

#성능 지표 시각화 
import matplotlib.pyplot as plt
import numpy as np

labels = ["BLEU", "Word\nOverlap", "Char\nOverlap", "Length\nRatio"]
values = [0.0274, 0.2644, 0.6389, 1.0081]  #성능 결과값 

# 각 축을 레이더에 맞게 0–1 (길이비율은 0.5–1.5) 정규화
norm_values = [
    values[0],                 # BLEU (0–1)
    values[1],                 # 단어 겹침 (0–1)
    values[2],                 # 문자 겹침 (0–1)
    (values[3]-0.5)/1.0        # 길이비율 0.5–1.5 → 정규화
]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
norm_values += norm_values[:1]  # 닫기 위해 처음 값 반복
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw=dict(polar=True))
ax.plot(angles, norm_values, 'o-', linewidth=2)
ax.fill(angles, norm_values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
plt.show()

# 처리 시간 비교 막대
fig, ax = plt.subplots(figsize=(6,2))
times = [397.97, 26.53]
labels = ["Total translation time (s)", "average time per sentence (s)"]
ax.barh(labels, times, color=["#777",'#444'])
for i, v in enumerate(times):
    ax.text(v + 5, i, f"{v:.1f}", va='center')
ax.set_xlim(0, max(times)*1.1)
plt.show()
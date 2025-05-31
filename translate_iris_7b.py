#라이브러리 Import
from transformers import AutoModelForCausalLM, AutoTokenizer   #허깅페이스 ai 모델 사용 도구
import torch
import time   #시간 측정
import re
from collections import Counter   #성능 평가 계산용
import math   #성능 평가 계산용
import matplotlib.pyplot as plt
import numpy as np

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

# 번역할 영어 문장
test_sentences_en = [
    "ok babe",
    "what happend babe",
    "oh man i hate that what kinda of god?",
    "ic i have a pitbull hes a sweet heart",
    "he thinks his a small dog",
    "how often do u shave ur legs?",
    "if he had his way he would but he will streach out and take over the couch",
    "u have hair down there?",
    "want to see his baby pics",
    "u ever have enough to shave it?",
    "woundering where u where and doing baby",
    "how long is ur hair?",
    "I know, Im excited to see u",
    "that would be nice, u ever kiss a boy?",
    "you know what baby your sssoooo allsome",
    "youll be a little heart breaker when your older",
    "yeah i ordered papjohns pizza and u baby",
    "didnt know you were only 13 but its ok",
    "ok babby what u eatting ?",
    "send me the sexy pic",
    "yah my day was good",
    "had a couple beers and was buzzing for a little bit",
    "it really turns me on.",
    "email me when u have time to get together...",
    "not married.. im divorced 5 years"
]

# 참조 번역
reference_translations = [
    "좋아, 자기야",                            # ok babe
    "무슨 일이야, 자기야?",                       # what happend babe
    "아, 이런 거 정말 싫어… 이런 걸 만든 신이 대체 누구야?",      # oh man i hate that what kinda of god?
    "알겠어, 나 핏불 키워. 정말 상냥한 녀석이야.",          # ic i have a pitbull hes a sweet heart
    "그는 자신이 작은 강아지라고 생각해.",            # he thinks his a small dog
    "다리 얼마나 자주 밀어?",                       # how often do u shave ur legs?
    "만약 마음대로 할 수 있다면, 그는 드러눕고, 소파를 다 차지할 거야.",    # if he had his way he would but he will streach out and take over the couch
    "거기 털 있어?",                            # u have hair down there?
    "그의 아기 때 사진 보고 싶어?",                       # want to see his baby pics
    "그걸 깎을 만큼 충분히 길었던 적 있어?",                 # u ever have enough to shave it?
    "네가 어디서 뭘 하고 있는지 궁금해, 자기야.",               # woundering where u where and doing baby
    "머리카락 길이가 얼마나 돼?",                    # how long is ur hair?
    "알아, 너 만나는 거 기대돼.",                    # I know, Im excited to see u
    "그러면 좋겠네요, 남자와 키스해 본 적 있나요?",                # that would be nice, u ever kiss a boy?
    "있잖아, 자기야, 넌 정말 대단해.",                   # you know what baby your sssoooo allsome
    "네가 크면 사람 마음 많이 아프게 할 거야.",             # youll be a little heart breaker when your older
    "응, 파파존스 피자를 주문했어… 그리고 네 생각 중이야, 자기야.",  # yeah i ordered papjohns pizza and u baby
    "당신이 겨우 13살인 줄 몰랐지만 괜찮습니다",               # didnt know you were only 13 but its ok  (미성년자 언급 → 빈칸)
    "오케이, 자기야, 뭐 먹고 있어?",                     # ok babby what u eatting ?
    "나한테 섹시한 사진 보내줘",                         # send me the sexy pic  (노골적 성적 요청 → 빈칸)
    "응, 내 하루는 괜찮았어.",                        # yah my day was good
    "맥주 몇 잔 마셨더니 잠깐 취했어.",                  # had a couple beers and was buzzing for a little bit
    "그거 정말 흥분돼",                                      # it really turns me on.  (노골적 성적 암시 → 빈칸)
    "시간 되면 모임 잡자고 이메일 줘.",                 # email me when u have time to get together...
    "결혼 안 했어.. 이혼한 지 5년 됐어."                   # not married.. im divorced 5 years
]

def calculate_bleu_score(reference, candidate):
    """BLEU 점수 계산 (1-gram ~ 4-gram)"""
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

            print(f"\n 성능 지표:")
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
            print(f"번역 중 오류 발생: {e}")
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
    avg_metrics = None
    
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

    # 🔥 성능 지표와 시간 정보를 반환
    return translated_results, avg_metrics, total_time

def create_performance_visualization(avg_metrics, total_time, num_sentences):
    """성능 지표 시각화 함수"""
    
    # 1. 레이더 차트 생성
    labels = ["BLEU", "Word\nOverlap", "Char\nOverlap", "Length\nRatio"]
    values = [
        avg_metrics['bleu_score'],
        avg_metrics['word_overlap'], 
        avg_metrics['char_overlap'],
        avg_metrics['length_ratio']
    ]

    # 각 축을 레이더에 맞게 0–1 (길이비율은 0.5–1.5) 정규화
    norm_values = [
        values[0],                      # BLEU (0–1)
        values[1],                      # 단어 겹침 (0–1)
        values[2],                      # 문자 겹침 (0–1)
        max(0, min(1, (values[3]-0.5)/1.0))  # 길이비율 0.5–1.5 → 정규화
    ]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    norm_values += norm_values[:1]  # 닫기 위해 처음 값 반복
    angles += angles[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 레이더 차트
    ax1 = plt.subplot(121, polar=True)
    ax1.plot(angles, norm_values, 'o-', linewidth=2, color='#2E86AB')
    ax1.fill(angles, norm_values, alpha=0.25, color='#2E86AB')
    ax1.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax1.set_ylim(0, 1)
    ax1.set_title('evaluation metrics', size=14, pad=20)
    
    # 각 지표의 실제 값을 텍스트로 표시
    for angle, value, label in zip(angles[:-1], values, labels):
        ax1.text(angle, 1.1, f'{value:.3f}', ha='center', va='center', fontsize=10)

    # 2. 시간 비교 막대 차트
    ax2 = plt.subplot(122)
    avg_time = total_time / num_sentences
    times = [total_time, avg_time]
    labels_time = ["Total time (s)", "Avg time per\nsentence (s)"]
    colors = ['#A23B72', '#F18F01']
    
    bars = ax2.bar(labels_time, times, color=colors)
    
    # 막대 위에 값 표시
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.2f}', ha='center', va='bottom', fontsize=11)
    
    ax2.set_ylabel('time(sec)')
    ax2.set_title('translation time', size=14)
    ax2.set_ylim(0, max(times)*1.15)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 성능 지표 요약 테이블 출력
    print("\n📊 성능 지표 요약")
    print("=" * 50)
    print(f"BLEU Score:     {avg_metrics['bleu_score']:.4f}")
    print(f"Word Overlap:   {avg_metrics['word_overlap']:.4f}")
    print(f"Char Overlap:   {avg_metrics['char_overlap']:.4f}")
    print(f"Length Ratio:   {avg_metrics['length_ratio']:.4f}")
    print(f"Total Time:     {total_time:.2f}초")
    print(f"Avg Time:       {total_time/num_sentences:.2f}초/문장")
    print("=" * 50)

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

    # 번역 실행 및 성능 지표 저장
    results, avg_metrics, total_time = translate_sentences()

    # 결과 출력
    save_results(results)

    print(f"\n 총 {len(results)}개 문장 번역 및 평가 완료!")
    
    # 🔥 자동으로 시각화 생성
    if avg_metrics is not None:
        print("\n성능 시각화를 생성합니다...")
        create_performance_visualization(avg_metrics, total_time, len(results))
    else:
        print("성능 지표를 계산할 수 없어서 시각화를 생성하지 못했습니다.")

if __name__ == "__main__":
    main()
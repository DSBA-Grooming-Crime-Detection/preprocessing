import xml.etree.ElementTree as ET

# 파일 경로
xml_file = "/content/drive/MyDrive/디스부_최종프로젝트/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
predator_file = "/content/drive/MyDrive/디스부_최종프로젝트/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

# Predator ID 불러오기
with open(predator_file, 'r') as f:
    predator_ids = set(line.strip() for line in f if line.strip())

# XML 파싱
tree = ET.parse(xml_file)
root = tree.getroot()

# Predator가 포함된 대화만 저장
conversations_with_predator = []

for conversation in root.findall('conversation'):
    conv_id = conversation.attrib['id']
    messages = []
    authors_in_convo = set()

    for message in conversation.findall('message'):
        author_el = message.find('author')
        time_el = message.find('time')
        text_el = message.find('text')

        # 요소가 없거나, 텍스트가 None이면 skip
        if author_el is None or time_el is None or text_el is None:
            continue

        author = author_el.text.strip() if author_el.text else ""
        time = time_el.text.strip() if time_el.text else ""
        text = text_el.text.strip() if text_el.text else ""

        messages.append({
            'author': author,
            'time': time,
            'text': text
        })

        authors_in_convo.add(author)

    # Predator가 있는 대화만 저장
    if any(author in predator_ids for author in authors_in_convo):
        conversations_with_predator.append({
            'conversation_id': conv_id,
            'messages': messages
        })

print(f"총 {len(conversations_with_predator)}개의 대화에서 predator가 발견되었습니다.")

"""Offline demonstration. No Telegram account or Ollama installation is needed."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from study_agent.agent import Agent
from study_agent.config import Config
from study_agent.store import Store


def vector(text: str) -> list[float]:
    words = text.lower().split()
    return [float(sum(word.startswith(letter) for word in words)) for letter in 'абвгдежзийклмнопрстуфхцчшщэюя']


class DemoTools:
    def __init__(self, store: Store):
        self.store = store
        self.actions: list[str] = []

    def call(self, name: str, **args: Any) -> Any:
        if name == 'is_processed':
            return {'processed': self.store.processed(args['chat_id'], args['message_id'], args['text'])}
        if name == 'get_source_task':
            return self.store.source_task(args['chat_id'], args['message_id'])
        if name == 'find_similar_tasks':
            return self.store.similar_tasks(vector(args['text']), args.get('subject'))
        if name == 'search_documents':
            return self.store.search_documents(vector(args['text']), args.get('subject'))
        if name == 'save_deadline':
            task_id = self.store.save_task(args['task_id'], args['subject'], args['title'], args['due_at'],
                                           args['chat_id'], args['message_id'], vector(args['subject'] + ' ' + args['title']), args['status'])
            return self.store.get_task(task_id)
        if name == 'mark_message':
            self.store.mark_message(args['chat_id'], args['message_id'], args['text'], args['event_type'], args['task_id'])
            return {'ok': True}
        if name == 'forward_to_topic':
            self.actions.append(f"Пересылка в топик: {args['subject']}")
            return {'message_id': len(self.actions)}
        if name == 'publish_deadlines':
            self.actions.append('Список дедлайнов обновлён')
            return {'edited': True}
        if name == 'list_deadlines':
            return self.store.deadlines()
        raise ValueError(name)


class DemoModel:
    def __init__(self):
        self.decisions = [
            {'action': 'create', 'subject': 'Математика', 'title': 'Домашняя работа № 3', 'due_at': '2026-11-05T23:59:00+03:00'},
            {'action': 'update', 'subject': 'Математика', 'title': 'Домашняя работа № 3', 'due_at': '2026-11-07T23:59:00+03:00', 'target_id': 1},
            {'action': 'ignore'},
        ]

    def decide(self, prompt: str) -> dict[str, Any]:
        return self.decisions.pop(0)


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        config = Config('demo', -100900, 99, frozenset({-100100}), {'Математика': 10, 'Программирование': 11},
                        {-100100: 'Математика'}, 'http://localhost:11434', 'qwen3:4b',
                        'nomic-embed-text', Path(folder) / 'demo.sqlite3')
        tools = DemoTools(Store(config.db_path))
        corpus = Path(__file__).parent / 'data' / 'sample_messages.jsonl'
        for line in corpus.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            tools.store.index_document(item['subject'], item['text'], vector(item['text']))
        print('Индексировано 12 учебных сообщений')
        agent = Agent(config, tools, DemoModel(), live=True)
        texts = [
            'По математике домашнюю работу № 3 сдать до 5 ноября, 23:59.',
            'Срок сдачи домашней работы № 3 по математике переносится на 7 ноября, 23:59.',
            'Кто сегодня идёт в столовую?',
        ]
        print('ДЕМО-РЕЖИМ: Telegram и LLM заменены сценарными имитациями')
        for number, item in enumerate(texts, 1):
            update = {'message': {'chat': {'id': -100100}, 'message_id': number, 'text': item}}
            result = agent.process(update)
            print(f'\nСообщение {number}: {item}')
            print('Решение:', result['action'])
            for action in tools.actions:
                print('Действие:', action)
            tools.actions.clear()
        print('\nИТОГОВЫЙ СПИСОК ДЕДЛАЙНОВ')
        for task in tools.call('list_deadlines'):
            print(f"{task['due_at']} | {task['subject']} | {task['title']}")
        tools.store.close()


if __name__ == '__main__':
    main()

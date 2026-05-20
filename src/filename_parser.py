import os
import re
from typing import List


class FileNameParser:
    """
    Парсер для извлечения признаков из имени файла.
    """

    @staticmethod
    def tokenize(filename: str) -> List[str]:
        # Удаляем расширение
        name = os.path.splitext(filename)[0]
        # Удаляем даты только если они явно отделены (например, _2023-12-01 или -2023-12-01)
        name = re.sub(r"([_\-.])\d{4}[-_.]?\d{2}[-_.]?\d{2}([_\-.]|$)", r"\1\2", name)
        # Разделяем по snake_case, camelCase, kebab-case, пробелам и точкам
        tokens = re.split(r"[ _\-.]+", name)
        # camelCase разбивка и объединение версий вида v2, v10
        camel_tokens = []
        for token in tokens:
            m = re.match(r"v\d+$", token, re.IGNORECASE)
            if m:
                camel_tokens.append(token)
            else:
                camel_tokens += re.findall(
                    r"[A-ZА-Я]?[a-zа-я]+|[A-ZА-Я]+(?![a-zа-я])|\d+", token
                )
        # Нормализация
        norm_tokens = [t.lower() for t in camel_tokens if t]
        return norm_tokens


if __name__ == "__main__":
    # Пример
    examples = [
        "Invoice_2023-12-01.pdf",
        "myProjectReport_v2.docx",
        "photo-20230411_123456.jpg",
        "archive_final-2022.zip",
        "README.md",
    ]
    for fname in examples:
        print(f"{fname} -> {FileNameParser.tokenize(fname)}")

from novel_reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_lds_test_from_chapters, read_ld_dev_from_chapters
from news_reader import read_ld_train, read_ld_test, read_ld_tests, read_ld_dev
import types

loader = types.SimpleNamespace()

loader.novel = types.SimpleNamespace()
loader.novel.train = read_ld_train_from_chapters
loader.novel.test = read_ld_test_from_chapters
loader.novel.dev = read_ld_dev_from_chapters

loader.news = types.SimpleNamespace()
loader.news.train = read_ld_train
loader.news.test = read_ld_test
loader.news.dev = read_ld_dev

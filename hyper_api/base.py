from datetime import datetime


class Base(object):
    def __repr__(self):
        return "{} : {} => {}".format(self.__class__.__name__, self.__getattribute__('name'), [x for x in dir(self) if "__" not in x])

    @staticmethod
    def str2date(string, date_format):
        if string is None:
            return None
        return datetime.strptime(string, date_format)

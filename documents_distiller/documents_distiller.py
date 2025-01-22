from typing import List, Any
from ..utils import LangchainOutputParser


class DocumentsDistiller:
    """
    A class designed to distill essential information from multiple documents into a combined
    structure, using natural language processing tools to extract and consolidate information.
    """
    def __init__(self, llm_model) -> None:
        """
        Initializes the DocumentsDistiller with specified language model
        
        Args:
        llm_model: The language model instance to be used for generating semantic blocks.
        """
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=None)

    
    @staticmethod
    def __combine_dicts(dict_list:List):
        """
        Combine a list of dictionaries into a single dictionary, merging values based on their types.
        
        Args:
        dict_list (List[dict]): A list of dictionaries to combine.
        
        Returns:
        dict: A combined dictionary with merged values.
        """
        combined_dict = {}
        
        for d in dict_list:
            for key, value in d.items():
                if key in combined_dict:
                    if isinstance(value, list) and isinstance(combined_dict[key], list):
                        combined_dict[key].extend(value)
                    elif isinstance(value, str) and isinstance(combined_dict[key], str):
                        if value and combined_dict[key]:
                            combined_dict[key] += f' {value}'
                        elif value:
                            combined_dict[key] = value
                    elif isinstance(value, dict) and isinstance(combined_dict[key], dict):
                        combined_dict[key].update(value)
                    else:
                        combined_dict[key] = value
                else:
                    combined_dict[key] = value
        
        return combined_dict, dict_list


    # 传字符串
    def distill(self, documents: str, output_data_structure, IE_query:str) -> tuple[dict[Any, str | Any], list]:
        """
        Distill information from multiple documents based on a specific information extraction query.
        
        Args:
        documents (List[str]): A list of documents from which to extract information.
        output_data_structure: The data structure definition for formatting the output JSON.
        IE_query (str): The query to provide to the language model for extracting information.
        
        Returns:
        dict: A dictionary representing distilled information from all documents.
        """

        # output_jsons = list(
        #     map(
        #         lambda context: self.langchain_output_parser.extract_information_as_json_for_context(
        #             context = context,
        #             IE_query=IE_query,
        #             output_data_structure= output_data_structure
        #             ),
        #         documents))

        output_json = self.langchain_output_parser.extract_information_as_json_for_context(
            context=documents,
            IE_query=IE_query,
            output_data_structure=output_data_structure
        )
        
        return DocumentsDistiller.__combine_dicts(output_json)


    # 传地址
    # def distill(self, documents: List[str], output_data_structure, IE_query: str) -> dict:
    #     """
    #     Distill information from multiple documents based on a specific information extraction query.
    #
    #     Args:
    #     documents (List[str]): A list of documents from which to extract information.
    #     output_data_structure: The data structure definition for formatting the output JSON.
    #     IE_query (str): The query to provide to the language model for extracting information.
    #
    #     Returns:
    #     dict: A dictionary representing distilled information from all documents.
    #     """
    #
    #     # 读取文件并整理格式
    #     def read_and_clean_file(file_path):
    #         try:
    #             with open(file_path, 'r', encoding='utf-8') as file:
    #                 content = file.read()
    #                 # 去除所有换行符
    #                 content_without_newlines = content.replace('\n', '').replace('\r', '')
    #                 return content_without_newlines
    #         except FileNotFoundError:
    #             print(f"文件未找到：{file_path}")
    #             return None
    #         except IOError:
    #             print(f"读取文件时发生错误：{file_path}")
    #             return None
    #
    #     output_jsons = list(
    #         map(
    #             lambda context: self.langchain_output_parser.extract_information_as_json_for_context(
    #                 context=read_and_clean_file(context),
    #                 IE_query=IE_query,
    #                 output_data_structure=output_data_structure
    #             ),
    #             documents))
    #
    #     return DocumentsDistiller.__combine_dicts(output_jsons)





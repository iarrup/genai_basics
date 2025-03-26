from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
"""

# class Revew(TypedDict):
#     key_themes: Annotated[list[str], "Write down all the key themes discussed in the review"]
#     summary: Annotated[str, "A brief summary of the review"]
#     sentiment: Annotated[str, "The sentiment of the review"]
#     pros: Annotated[list[str], "Write down all the pros metioned in the review"]
#     cons: Annotated[list[str], "Write down all the cons mentioned in the review"]
#     name: Annotated[Optional[str], "Write the name of the person who wrote the review"]


class Revew(BaseModel):
    key_themes: list[str] = Field(description = "Write down all the key themes discussed in the review")
    summary: str = Field(description = "A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description = "The sentiment of the review, either positive or negative")
    pros: Optional[list[str]] = Field(description = "Write down all the pros metioned in the review")
    cons: Optional[list[str]] = Field(description = "Write down all the cons mentioned in the review")
    name: Optional[str] = Field(description = "Write the name of the person who wrote the review")




model = ChatOpenAI(model_name = "gpt-4o-mini")
model_so = model.with_structured_output(Revew)

result = model_so.invoke(review)
print(result)


import express from "express";
const app=express();
import dotenv from "dotenv";
dotenv.config();
import cors from "cors";
import { pipeline } from '@xenova/transformers';
import { ChatGroq } from "@langchain/groq";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { QdrantClient } from "@qdrant/js-client-rest";
app.use(cors());
app.use(express.json());
const PORT=process.env.PORT||8000;
process.env.HF_TOKEN=process.env.HF_TOKEN;
const model=await pipeline('feature-extraction',
     'Xenova/all-MiniLM-L6-v2',  
);

async function embedQuery(text) {
    const result = await model(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
}

async function embedDocuments(texts) {
    const results = [];
    for (const text of texts) {
      const res = await model(text, { pooling: 'mean', normalize: true });
      results.push(Array.from(res.data));
    }
    return results;
}

const llm=new ChatGroq({
    model:"llama-3.3-70b-versatile",
    temperature:0.0,
    apiKey:process.env.LLM_APIKEY,
});

const client=new QdrantClient({
    url:"https://6b494f89-6031-447d-8ced-514c531c3b14.us-east-1-1.aws.cloud.qdrant.io",
    apiKey:process.env.QDRANT_APIKEY,
});

const anika=await client.getCollections();
console.log(anika);
const answer=z.object({
    imp_words: z.string().describe("List the important legal terms or keywords that a Judiciary aspirant should remember from the context provided to you."),
    description: z.string().describe("Provide a descriptive explanation based on the given context and query being asked."),
    exact: z.string().describe("Quote the exact text from the act or code, which you can derive from the context that will be provided to you.")
});

const structuredLlm=llm.withStructuredOutput(answer);
async function generate(query,act){
    try{
    const queryVector=await embedQuery(query);
    const searchResult=await client.search(`${act}`,{
        vector:queryVector,
        limit:3,
    });
    const pay=searchResult.map((result)=>["system", result.payload.text]);
    const prompt=ChatPromptTemplate.fromMessages([
    ["system",`You are an AI assistant which is designed
      to help law students who are preparing for Judiciary exams in India.
      You will be provided with context of the given act or code and according to the given query, you need to answer in a structured way, the structure will be already provided to you.
      The structure will have three section-: 1.) imp_words-: the important words from the provided context and in relation to the query asked. 2.) description-: this section must have a long descriptive answer to the user's query.
      3.) exact-: exact words as it is penned down in the code or act. Their maybe queries which are not related to the document so answer them accordingly.
      `],
    ...pay,
    ["human",`${query}`],
    ]);
    const chain=prompt.pipe(structuredLlm);
    const response=await chain.invoke({input:`${query}`});
    return {type:"success",content:response};
   }
   catch(e){
       return {type:"error",content:e};
   }
}

app.get("/",(req,res)=>{
    res.status(200).send("Do you think you have got the bullocks: William Butcher");
});

app.post("/:act",async(req,res)=>{
    const act=req.params.act;
    const query=req.body.query;
    const answer=await generate(query,act);
    console.log(answer);
    if(answer.type==="success"){
      res.status(200).json({type:"success",content:answer.content});
    }
    else{
      res.status(500).json({type:"error",content:answer.content});
    }
})


export default app;


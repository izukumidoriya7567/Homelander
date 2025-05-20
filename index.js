import express from "express";
const app=express();
import dotenv from "dotenv";
dotenv.config();
import cors from "cors";
import { pipeline } from '@xenova/transformers';
import {ChatGroq} from "@langchain/groq";
import {z} from "zod";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {QdrantClient} from "@qdrant/js-client-rest";
app.use(cors());
const PORT=process.env.PORT||8000;
const model=await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
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
    url:"https://d2abd0c8-b572-452f-9670-c776a382be87.us-east4-0.gcp.cloud.qdrant.io",
    apiKey:process.env.QDRANT_APIKEY,
});

const answer=z.object({
    imp_words: z.string().describe("List the important legal terms or keywords that a Judiciary aspirant should remember from this context."),
    description: z.string().describe("Provide a descriptive explanation based on the given context, relevant to the question asked."),
    exact: z.string().describe("Quote the exact text from the context that directly answers the question.")
});

const structuredLlm=llm.withStructuredOutput(answer);
async function generate(query,act){
    try{
    const queryVector=await embedQuery(query);
    const searchResult=await client.search(`${act}`,{
        vector:queryVector,
        limit:5,
    });
    const pay=searchResult.map((result)=>["system", result.payload.text]);
    const prompt=ChatPromptTemplate.fromMessages([
    ["system",`You are an AI assistant which is designed
      to help my sister Anika Tripathi who is a Law Student and wants to prepare for Judiciary exams in India.
      You will be provided with context, and you need to answer accordingly in a structured way, the structure is already provided to you.
      So, you just need to answer the questions based on the context, structure and the question provided to you. 
      If the question provided is out of context, please ask the user to put relevant questions in the Box.`],
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
})
app.get("/:act/:query",async(req,res)=>{
    const act=req.params.act;
    const query=req.params.query;
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
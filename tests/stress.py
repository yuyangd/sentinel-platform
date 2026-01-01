import aiohttp
import asyncio
import time

NLB_URL = "http://a62d0fc501b8743ca8f2ec0d334108a0-a0eec30244505014.elb.ap-southeast-2.amazonaws.com:8000/recommend"

async def ask_sentinel(session, req_id):
    payload = {"user_id": 42, "movie_ids": [100, 200, 300]}
    try:
        async with session.post(NLB_URL, json=payload) as response:
            result = await response.json()
            return response.status
    except Exception as e:
        return str(e)

async def swarm(num_requests=200, concurrency=20):
    print(f"🚀 Launching {num_requests} requests (Concurrency: {concurrency})...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        for i in range(num_requests):
            tasks.append(ask_sentinel(session, i))
            # Control concurrency batching
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)
            
        duration = time.time() - start_time
        print(f"✅ Done in {duration:.2f}s")
        print(f"⚡ RPS: {num_requests / duration:.2f}")

if __name__ == "__main__":
    # We send enough traffic to overwhelm 1 replica (which handles ~5 concurrent)
    asyncio.run(swarm(num_requests=500, concurrency=50))

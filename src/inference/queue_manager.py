"""
Request Queue Manager Module

Manages async request queue for batching and concurrent processing
"""
from asyncio import Queue as AsyncQueue
from typing import Dict, List, Optional, Callable, Awaitable
import asyncio
from loguru import logger
import time


class RequestQueueManager:
    """Manage request queue for batching"""
    
    def __init__(self, max_size: int = 1000, batch_timeout: float = 0.1, batch_size: int = 32):
        self.queue = AsyncQueue(maxsize=max_size)
        self.batch_timeout = batch_timeout
        self.batch_size = batch_size
        self.processing = False
        self._processor_task: Optional[asyncio.Task] = None
        
        logger.info(f"RequestQueueManager initialized: max_size={max_size}, batch_timeout={batch_timeout}, batch_size={batch_size}")
    
    async def enqueue(self, request: Dict) -> Dict:
        """Enqueue request and wait for result"""
        result_queue = AsyncQueue()
        request['result_queue'] = result_queue
        request['enqueue_time'] = time.time()
        
        try:
            await asyncio.wait_for(
                self.queue.put(request),
                timeout=5.0  # Max wait time to enqueue
            )
        except asyncio.TimeoutError:
            logger.warning("Queue full, request dropped")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'error': 'queue_full',
                'processing_time_ms': 0.0
            }
        
        # Wait for result
        try:
            result = await asyncio.wait_for(
                result_queue.get(),
                timeout=10.0  # Max wait time for result
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Result wait timed out")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'error': 'result_timeout',
                'processing_time_ms': 0.0
            }
    
    async def process_queue(
        self,
        processor: Callable[[List[Dict]], Awaitable[List[Dict]]]
    ):
        """Process queue continuously"""
        self.processing = True
        logger.info("Starting queue processor")
        
        while self.processing:
            batch = []
            batch_start = time.time()
            
            # Collect batch
            while len(batch) < self.batch_size:
                try:
                    # Wait for request with timeout
                    timeout = self.batch_timeout - (time.time() - batch_start)
                    if timeout <= 0:
                        break
                    
                    request = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=timeout
                    )
                    batch.append(request)
                    
                except asyncio.TimeoutError:
                    break
            
            # Process batch if we have requests
            if batch:
                await self._process_batch(batch, processor)
            else:
                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.01)
        
        logger.info("Queue processor stopped")
    
    async def _process_batch(
        self,
        batch: List[Dict],
        processor: Callable[[List[Dict]], Awaitable[List[Dict]]]
    ):
        """Process batch of requests"""
        try:
            # Extract request data (without result_queue)
            requests = []
            for req in batch:
                req_copy = {k: v for k, v in req.items() if k != 'result_queue' and k != 'enqueue_time'}
                requests.append(req_copy)
            
            # Process batch
            start_time = time.time()
            results = await processor(requests)
            processing_time = (time.time() - start_time) * 1000
            
            # Add processing time to results
            for i, result in enumerate(results):
                if 'processing_time_ms' not in result:
                    result['processing_time_ms'] = processing_time / len(batch)
            
            # Send results back
            for req, result in zip(batch, results):
                if 'result_queue' in req:
                    await req['result_queue'].put(result)
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Send error results
            for req in batch:
                if 'result_queue' in req:
                    await req['result_queue'].put({
                        'anomaly_score': 0.0,
                        'is_anomaly': False,
                        'error': str(e),
                        'processing_time_ms': 0.0
                    })
    
    def start_processing(
        self,
        processor: Callable[[List[Dict]], Awaitable[List[Dict]]]
    ):
        """Start processing queue in background"""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self.process_queue(processor))
            logger.info("Queue processor task started")
    
    async def stop(self):
        """Stop processing"""
        self.processing = False
        if self._processor_task and not self._processor_task.done():
            await self._processor_task
        logger.info("Queue processor stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()

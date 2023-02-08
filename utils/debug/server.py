from .base import Debugger, DebugFrame
from typing import  Optional, Generic, TypeVar
from queue import Full, Empty
from multiprocessing import Process, Manager, cpu_count
import asyncio
from asyncio.queues import QueueFull, QueueEmpty
T = TypeVar('T')

class RFPose:
	def __init__(self, type: str, id: Optional[int], translation: list[float], rotation: list[float]):
		self.type = type
		if id is not None:
			self.id = id
		self.translation = translation
		self.rotation = rotation

class GameItem:
	def __init__(self, type: str, id: Optional[int], poses: list[RFPose]) -> None:
		self.type = type
		if id is not None:
			self.id = id
		self.poses = poses

class ProcQueue(Generic[T]):
	def __init__(self, maxsize: int = 0):
		m = Manager()
		self._queue = m.Queue(maxsize=maxsize)
		self._real_executor = None
		self._cancelled_join = False

	@property
	def _executor(self):
		from concurrent.futures import ThreadPoolExecutor
		if not self._real_executor:
			self._real_executor = ThreadPoolExecutor(max_workers=cpu_count())
		return self._real_executor

	def __getstate__(self):
		self_dict = self.__dict__
		self_dict['_real_executor'] = None
		return self_dict

	def __getattr__(self, name):
		if name in { 'qsize', 'empty', 'full', 'put', 'put_nowait', 'get', 'get_nowait', 'close' }:
			return getattr(self._queue, name)
		else:
			raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
	
	def get(self, timeout: Optional[float] = None):
		return self._queue.get(timeout=timeout)
	
	def put(self, item: T, timeout: Optional[float] = None):
		return self._queue.put(item, timeout=timeout)

	async def put_async(self, item: T, *, timeout: Optional[float] = None):
		loop = asyncio.get_event_loop()
		try:
			return (await loop.run_in_executor(self._executor, self.put, item, timeout))
		except Full:
			raise QueueFull()

	async def get_async(self, timeout: Optional[float] = None) -> T:
		loop = asyncio.get_event_loop()
		try:
			return await loop.run_in_executor(self._executor, self.get, timeout)
		except Empty:
			raise QueueEmpty()

	def cancel_join_thread(self):
		self._cancelled_join = True
		self._queue.cancel_join_thread()

	def join_thread(self):
		self._queue.join_thread()
		if self._real_executor and not self._cancelled_join:
			self._real_executor.shutdown()

def _build_app(queue: ProcQueue[list[GameItem]]):
	from pathlib import Path
	from fastapi import FastAPI, Request
	from fastapi.staticfiles import StaticFiles
	from sse_starlette.sse import EventSourceResponse
	import json
	app = FastAPI()

	api_app = FastAPI(title="my existing api")
	@api_app.get('/data')
	async def message_stream(request: Request):
		async def _event_generator():
			while True:
				# If client closes connection, stop sending events
				if await request.is_disconnected():
					break

				# Checks for new messages and return them to client if any
				try:
					data = await queue.get_async(timeout=.1)
				except QueueEmpty:
					continue

				yield {
					'event': 'pose_update',
					'id': '',
					'retry': 1,
					'data': json.dumps(data, default=lambda x: x.__dict__)
				}
		return EventSourceResponse(_event_generator())
	
	app.mount('/api', api_app)
	app.mount('/', StaticFiles(directory=Path(__file__).parent / "site/dist", html=True), name="static")
	return app

def _run_webserver(queue: ProcQueue[list[GameItem]], port: int):
	app = _build_app(queue)
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=port)


class WebDebug(Debugger):
	def __init__(self, port: int = 8081):
		super().__init__()
		self._wsdata: ProcQueue[list[GameItem]] = ProcQueue(2)
		# _run_webserver(self._wsdata)
		self._wsproc = Process(
			target=_run_webserver,
			name='webserver',
			args=(self._wsdata, port),
			daemon=True,
		)
		self._wsproc.start()
	
	def __del__(self):
		# Kill subprocess when we go out of scope
		if getattr(self, '_wsproc', None) is not None:
			self._wsproc.kill()

	def finish_frame(self, frame: DebugFrame):
		ws_data = [
			GameItem(
				item_id.type,
				getattr(item_id, 'id', None),
				[
					RFPose(
						rf_id.type,
						getattr(rf_id, 'id', None),
						list(pose[0]),
						list(pose[1]),
					)
					for rf_id, pose in poses.items()
				],
			)
			for item_id, poses in frame._records.items()
		]
		try:
			self._wsdata.put(ws_data, timeout=.05)
		except Full:
			# If the queue is full, don't try adding more
			pass
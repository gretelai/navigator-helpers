# 🧱 Building New Pipelines

To build a new pipeline, you will need to do the following:

### **1. Create a task suite.**

A task suite is a collection of tasks that are associated with a particular use case (e.g., text-to-code). Each task is implemented as a method in a `TaskSuite` class. Tasks can be composed of multiple sub-tasks. For example, `create_record` might execute all the tasks associated with the creation of a single record.

To build your own task suite, create a class that inherits from [BaseTaskSuite](../tasks/base.py) and implement the tasks for your use case. For example:

```python
from navigator_helpers.tasks.base import BaseTaskSuite

class MyTaskSuite(BaseTaskSuite):

    def task_1(self, ...):
        # the base class provides access to the LLM suite
        self.llm_suite

    def task_2(self, ...):
        ...

    def task_3(self, ...):
        ...
```

### **2. Create a pipeline configuration.**

Pipeline configs are pydantic models. Create a new config by inheriting from [BasePipelineConfig](config/base.py) and defining the fields that your pipeline needs. For example:

```python
from pydantic import BaseModel

from navigator_helpers.pipelines.config.base import PipelineConfig

class MyPipeConfig(PipelineConfig, BaseModel):
    super_important_parameter: str
```

Next, add logic to the helper [smart_load_pipeline_config](config/utils.py) function to streamline the loading of your pipeline config. This function makes it possible to pass in your config as a path, yaml string, or dictionary.

### **3. Create a pipeline.**

A pipeline is a collection of tasks that are executed in a specific order. To build your own pipeline, create a class that inherits from [BasePipeline](base.py) and implements the `tasks` property and `run` method. For example:

```python
from navigator_helpers.llms.llm_suite import GretelLLMSuite
from navigator_helpers.pipelines.base import BasePipeline, PipelineResults

class MyPipeline(BasePipeline):

    @property
    def tasks(self):
        # self._tasks is set to None in the base class
        if self._tasks is None:
            self._tasks = MyTaskSuite(GretelLLMSuite())
        return self._tasks

    def setup_pipeline(self):
        # optionally create this method for additional setting up
        # it is run at the end of the constructor in the base class
        ...

    def run(self, config: MyPipeConfig) -> PipelineResults:
        # run the tasks in the order you want
        self.tasks.task_1(...)
        self.tasks.task_2(...)
        self.tasks.task_3(...)
        return PipelineResults(dataframe=df, config=config, metadata=metadata)

config_str = "super_important_parameter: 'the value is 42'"
pipe = MyPipeline(config_str)

results = pipe.run()
```


.. _scenedetect-scene_list:

----------------------------------------------
Scene & SceneList
----------------------------------------------

.. automodule:: scenedetect.scene_list


=========================================
Usage Examples
=========================================

:py:class:`SceneList` can be treated as a list of :py:class:`Scene` objects, but with extended
methods to transform the list of scenes. For example:

.. code:: python

    scenes = sm.get_scene_list()
    new_scenes = scenes.merge_short_scenes(20, 10)
    for i, scene in enumerate(processed):
        # scene.start == scene[0], scene.end == scene[1]
        print('Scene {scene_num}: start = {start}, end = {end}\n'.format(
            scene_num = i, start = scene.start, end = scene.end
        ))

Operations can also be chained together:

    new_scenes = scenes
        .merge_short_scenes(min_scene_len=10, merge_len=5)
        .drop_short_scenes(min_scene_len=10)
        .shift_scenes(shift_end=5)

:py:class:`Scene` objects themselves are tuples of two :py:class:`FrameTimecode`s representing the
`start` and `end` of each scene, but with named accessors:

.. code:: python

    # Assuming `scenes` is a SceneList or list of Scenes:
    for scene in scenes:
        assert scene.start == scene[0]
        assert scene.end == scene[1]


``Scene`` Class
=========================================

.. autoclass:: scenedetect.scene_list.Scene
   :members:
   :undoc-members:


``SceneList`` Class
=========================================

.. autoclass:: scenedetect.scene_list.SceneList
   :members:
   :undoc-members:



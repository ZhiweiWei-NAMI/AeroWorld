using UnrealBuildTool;

public class AeroEditorTools : ModuleRules
{
	public AeroEditorTools(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
				"Core"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new[]
			{
				"AssetRegistry",
				"AssetTools",
				"BlueprintEditorLibrary",
				"CoreUObject",
				"EditorSubsystem",
				"Engine",
				"Json",
				"JsonUtilities",
				"Kismet",
				"PedestrianRuntime",
				"Projects",
				"UnrealEd",
				"AeroPedNavSemantic",
				"AeroSemanticRuntime",
				"GeometryScriptingCore",
				"GeometryCore",
				"DynamicMesh",
				"GeometryFramework"
			}
		);
	}
}

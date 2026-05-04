#include "SumoImporterEditorModule.h"

#include "DesktopPlatformModule.h"
#include "CityRoadGeoJsonParser.h"
#include "Editor.h"
#include "Framework/Application/SlateApplication.h"
#include "IDesktopPlatform.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/MessageDialog.h"
#include "Misc/Paths.h"
#include "Styling/AppStyle.h"
#include "SumoImporterLog.h"
#include "SumoNetParser.h"
#include "SumoRoadNetworkActor.h"
#include "SumoSceneBuilder.h"
#include "ToolMenus.h"

#define LOCTEXT_NAMESPACE "FSumoImporterEditorModule"

namespace
{
bool IsNetXmlFile(const FString& FilePath)
{
	return FPaths::GetExtension(FilePath, true).Equals(TEXT(".net.xml"), ESearchCase::IgnoreCase);
}

bool IsGeoJsonFile(const FString& FilePath)
{
	return FPaths::GetExtension(FilePath, true).Equals(TEXT(".geojson"), ESearchCase::IgnoreCase);
}

bool IsBoundsGeoJsonFile(const FString& FilePath)
{
	return FPaths::GetBaseFilename(FilePath).Equals(TEXT("bounds"), ESearchCase::IgnoreCase);
}

FString SelectRoadGeoJsonFile(const TArray<FString>& GeoJsonFiles)
{
	if (GeoJsonFiles.IsEmpty())
	{
		return FString();
	}

	for (const FString& GeoFile : GeoJsonFiles)
	{
		const FString BaseName = FPaths::GetBaseFilename(GeoFile).ToLower();
		if (BaseName.Contains(TEXT("road")) && !IsBoundsGeoJsonFile(GeoFile))
		{
			return GeoFile;
		}
	}

	for (const FString& GeoFile : GeoJsonFiles)
	{
		if (!IsBoundsGeoJsonFile(GeoFile))
		{
			return GeoFile;
		}
	}

	return FString();
}

FString SelectBoundsGeoJsonFile(const TArray<FString>& GeoJsonFiles)
{
	for (const FString& GeoFile : GeoJsonFiles)
	{
		if (IsBoundsGeoJsonFile(GeoFile))
		{
			return GeoFile;
		}
	}
	return FString();
}
}

void FSumoImporterEditorModule::StartupModule()
{
	UToolMenus::RegisterStartupCallback(
		FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FSumoImporterEditorModule::RegisterMenus));
}

void FSumoImporterEditorModule::ShutdownModule()
{
	if (UToolMenus::IsToolMenuUIEnabled())
	{
		UToolMenus::UnRegisterStartupCallback(this);
		UToolMenus::UnregisterOwner(this);
	}
}

void FSumoImporterEditorModule::RegisterMenus()
{
	FToolMenuOwnerScoped OwnerScoped(this);

	UToolMenu* ToolbarMenu = UToolMenus::Get()->ExtendMenu(TEXT("LevelEditor.LevelEditorToolBar"));
	FToolMenuSection& Section = ToolbarMenu->FindOrAddSection(TEXT("SumoImporter"));
	Section.AddEntry(FToolMenuEntry::InitToolBarButton(
		TEXT("SumoImporter.ImportNetXml"),
		FUIAction(FExecuteAction::CreateRaw(this, &FSumoImporterEditorModule::HandleImportNetXmlClicked)),
		LOCTEXT("ImportNetXmlLabel", "Import SUMO Net"),
		LOCTEXT("ImportNetXmlTooltip", "Read a SUMO net.xml and generate lane splines + junction debug geometry."),
		FSlateIcon(FAppStyle::GetAppStyleSetName(), TEXT("Icons.Import"))));

	UToolMenu* ToolsMenu = UToolMenus::Get()->ExtendMenu(TEXT("LevelEditor.MainMenu.Tools"));
	FToolMenuSection& ToolsSection = ToolsMenu->FindOrAddSection(TEXT("SumoImporter"));
	ToolsSection.AddMenuEntry(
		TEXT("SumoImporter.ImportNetXmlMenu"),
		LOCTEXT("ImportNetXmlMenuLabel", "Import SUMO Net"),
		LOCTEXT("ImportNetXmlMenuTooltip", "Read a SUMO net.xml and generate lane splines + junction debug geometry."),
		FSlateIcon(FAppStyle::GetAppStyleSetName(), TEXT("Icons.Import")),
		FUIAction(FExecuteAction::CreateRaw(this, &FSumoImporterEditorModule::HandleImportNetXmlClicked)));

	UE_LOG(LogSumoImporter, Log, TEXT("SumoImporterEditor menus registered."));
}

void FSumoImporterEditorModule::HandleImportNetXmlClicked()
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform == nullptr)
	{
		ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("DesktopPlatform is unavailable."), true);
		return;
	}

	FString DefaultDirectory = FPaths::ProjectDir();
	if (const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("SumoImporter")))
	{
		DefaultDirectory = Plugin->GetBaseDir();
	}

	TArray<FString> ChosenFiles;
	const bool bOpened = DesktopPlatform->OpenFileDialog(
		FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
		TEXT("Select SUMO net.xml OR City roads.geojson (+ optional bounds.geojson)"),
		DefaultDirectory,
		TEXT("map.net.xml"),
		TEXT("Road Network Files (*.net.xml;*.geojson)|*.net.xml;*.geojson|SUMO Net XML (*.net.xml)|*.net.xml|GeoJSON (*.geojson)|*.geojson|All Files (*.*)|*.*"),
		EFileDialogFlags::Multiple,
		ChosenFiles);

	if (!bOpened || ChosenFiles.Num() == 0)
	{
		return;
	}

	UWorld* EditorWorld = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
	if (EditorWorld == nullptr)
	{
		ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("Editor world is unavailable."), true);
		return;
	}

	TArray<FString> NetXmlFiles;
	TArray<FString> GeoJsonFiles;
	for (const FString& FilePath : ChosenFiles)
	{
		if (IsNetXmlFile(FilePath))
		{
			NetXmlFiles.Add(FilePath);
		}
		else if (IsGeoJsonFile(FilePath))
		{
			GeoJsonFiles.Add(FilePath);
		}
	}

	if (!NetXmlFiles.IsEmpty() && !GeoJsonFiles.IsEmpty())
	{
		ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("Please select either net.xml OR geojson files, not both in one import."), true);
		return;
	}

	if (NetXmlFiles.Num() > 1)
	{
		ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("Please select only one net.xml file."), true);
		return;
	}

	FString InputFile;
	FString BoundsGeoJsonFile;
	bool bIsGeoJson = false;
	if (!GeoJsonFiles.IsEmpty())
	{
		bIsGeoJson = true;
		InputFile = SelectRoadGeoJsonFile(GeoJsonFiles);
		BoundsGeoJsonFile = SelectBoundsGeoJsonFile(GeoJsonFiles);
		if (InputFile.IsEmpty())
		{
			ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("No valid roads.geojson found. Please select road.geojson (and optionally bounds.geojson)."), true);
			return;
		}
	}
	else if (!NetXmlFiles.IsEmpty())
	{
		InputFile = NetXmlFiles[0];
	}
	else
	{
		ShowResultDialog(TEXT("SUMO Import Failed"), TEXT("No supported file type selected."), true);
		return;
	}

	FSumoNetData NetData;
	FSumoImportStats Stats;
	FString Error;

	bool bParseResult = false;
	if (bIsGeoJson)
	{
		bParseResult = FCityRoadGeoJsonParser::ParseFile(InputFile, BoundsGeoJsonFile, NetData, Stats, Error);
	}
	else
	{
		FSumoParseOptions ParseOptions;
		ParseOptions.bImportInternalEdges = false;
		bParseResult = FSumoNetParser::ParseFile(InputFile, ParseOptions, NetData, Stats, Error);
	}

	if (!bParseResult)
	{
		UE_LOG(LogSumoImporter, Error, TEXT("Parse failed: %s"), *Error);
		ShowResultDialog(TEXT("SUMO Import Failed"), Error, true);
		return;
	}

	FSumoTransformConfig TransformConfig;
	if (bIsGeoJson)
	{
		TransformConfig.AxisMapping = ESumoAxisMapping::XY_To_XNegY;
		TransformConfig.YawOffsetDeg = 180.0f;
		UE_LOG(
			LogSumoImporter,
			Log,
			TEXT("GeoJSON transform config: AxisMapping=XY_To_XNegY, YawOffsetDeg=%.1f"),
			TransformConfig.YawOffsetDeg);
	}

	FSumoBuildOptions BuildOptions;
	BuildOptions.bReplaceExistingActor = true;
	BuildOptions.bBuildJunctionDebug = true;
	BuildOptions.NetworkActorName = TEXT("SUMO_RoadNetwork");

	ASumoRoadNetworkActor* NetworkActor = FSumoSceneBuilder::BuildToWorld(
		EditorWorld,
		NetData,
		TransformConfig,
		BuildOptions,
		Stats,
		Error);

	if (!IsValid(NetworkActor))
	{
		UE_LOG(LogSumoImporter, Error, TEXT("Build failed: %s"), *Error);
		ShowResultDialog(TEXT("SUMO Import Failed"), Error, true);
		return;
	}

	EditorWorld->MarkPackageDirty();
	NetworkActor->MarkPackageDirty();

	const FString Summary = FString::Printf(
		TEXT("Import complete.\nType: %s\nFile: %s\nImportedEdges: %d\nImportedLanes: %d\nJunctions: %d\nConnections: %d\nSkippedEdges: %d\nSkippedLanes: %d\nWarnings: %d"),
		bIsGeoJson ? TEXT("CityGenerator GeoJSON") : TEXT("SUMO net.xml"),
		*FPaths::GetCleanFilename(InputFile),
		Stats.ImportedEdges,
		Stats.ImportedLanes,
		Stats.JunctionCount,
		Stats.ConnectionCount,
		Stats.SkippedEdges,
		Stats.SkippedLanes,
		Stats.WarningCount);

	UE_LOG(LogSumoImporter, Log, TEXT("%s"), *Summary);
	ShowResultDialog(TEXT("SUMO Import"), Summary, false);
}

void FSumoImporterEditorModule::ShowResultDialog(const FString& Title, const FString& Message, bool bIsError) const
{
	const FText DisplayText = FText::FromString(Title + TEXT("\n\n") + Message);
	const EAppMsgType::Type DialogType = EAppMsgType::Ok;
	FMessageDialog::Open(DialogType, DisplayText);

	if (bIsError)
	{
		UE_LOG(LogSumoImporter, Error, TEXT("%s"), *Message);
	}
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FSumoImporterEditorModule, SumoImporterEditor)

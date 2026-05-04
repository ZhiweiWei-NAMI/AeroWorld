// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "IStructureDetailsView.h"
#include "IDetailCustomization.h"
class FToolBarBuilder;
class FMenuBuilder;
class SOverlay;
class SWebBrowserView;
struct FBrowserContextSettings;
class IDetailCustomization;
namespace TemplateParameter { class SDockTab; }

class FTwinFabricEditorModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
	
	/** This function will be bound to Command. */
	void PluginButtonClicked();
	
private:

	void RegisterMenus();
	TSharedRef<class SDockTab> OnSpawnPluginTab(const class FSpawnTabArgs& SpawnTabArgs);
	void OnTabClosed(TSharedRef<SDockTab> DockTab);
	void CreateDetailView();

private:
	TSharedPtr<class FUICommandList> PluginCommands;
	TArray<FString> Tips = TArray<FString>();
	TSharedPtr<SOverlay> BrowserContainer;
	TSharedPtr<SWebBrowserView> BrowserView;
	TSharedPtr<FBrowserContextSettings> BrowserSettings;
	TSharedPtr<IStructureDetailsView> SettingsView;
};

class FSettingsDetails : public IDetailCustomization
{
public:
	/** Makes a new instance of this detail layout class for a specific detail view requesting it */
	static TSharedRef<IDetailCustomization> MakeInstance()
	{
		return MakeShareable(new FSettingsDetails());
	}

	/** IDetailCustomization interface */
	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
};
